package ai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/invopop/jsonschema"
	"github.com/sashabaranov/go-openai"
)

func newTool(
	name, description string,
	structSchema any,
) openai.Tool {
	reflector := jsonschema.Reflector{
		DoNotReference: true,
		Anonymous:      true,
	}
	schema := reflector.Reflect(structSchema)
	schema.Version = ""
	schema.AdditionalProperties = nil

	parameters, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		// this should not be possible
		panic(err)
	}

	return openai.Tool{
		Type: "function",
		Function: &openai.FunctionDefinition{
			Name:        name,
			Description: description,
			Parameters:  json.RawMessage(parameters),
		},
	}
}

type Tool interface {
	Name() string
	Description() string
	Parameters() any
	Execute(string) (string, error)
}

func NewTool[T any](name, description string, execute func(T) (any, error)) Tool {
	return tool[T]{name, description, execute}
}

type tool[T any] struct {
	name        string
	description string
	execute     func(T) (any, error)
}

func (t tool[T]) Name() string {
	return t.name
}

func (t tool[T]) Description() string {
	return t.description
}

func (t tool[T]) Parameters() any {
	var v T
	return v
}

func (t tool[T]) Execute(payload string) (string, error) {
	var args T
	if err := json.Unmarshal([]byte(payload), &args); err != nil {
		return "", err
	}

	res, err := t.execute(args)
	if err != nil {
		return "", err
	}

	bs, err := json.Marshal(res)
	if err != nil {
		return "", err
	}

	return string(bs), nil
}

type TextGenerator struct {
	Client      *openai.Client
	Model       string
	System      string
	Prompt      string
	Temperature float32
	Tools       []Tool
}

type Generation struct {
	Text       string
	Roundtrips int
	Messages   []openai.ChatCompletionMessage
}

func (g TextGenerator) Generate(ctx context.Context) (*Generation, error) {
	var tools []openai.Tool
	for _, tool := range g.Tools {
		tools = append(tools, newTool(tool.Name(), tool.Description(), tool.Parameters()))
	}

	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: g.System,
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: g.Prompt,
		},
	}

	roundtripLimit := 10

	for i := 0; i < roundtripLimit; i++ {
		request := openai.ChatCompletionRequest{
			Model:       g.Model,
			Temperature: g.Temperature,
			Messages:    messages,
			Tools:       tools,
		}

		resp, err := g.Client.CreateChatCompletion(ctx, request)
		if err != nil {
			return nil, fmt.Errorf("chat completion error: %w", err)
		}

		assistantMessage := resp.Choices[0].Message
		messages = append(messages, assistantMessage)

		if len(assistantMessage.ToolCalls) > 0 {
			for _, toolCall := range assistantMessage.ToolCalls {
				var foundTool Tool
				for _, tool := range g.Tools {
					if tool.Name() == toolCall.Function.Name {
						foundTool = tool
						goto didFindTool
					}
				}
				return nil, fmt.Errorf("tool %v not found", toolCall.Function.Name)

			didFindTool:

				res, err := foundTool.Execute(toolCall.Function.Arguments)
				if err != nil {
					return nil, fmt.Errorf("tool %v failed: %w", toolCall.Function.Name, err)
				}

				messages = append(messages, openai.ChatCompletionMessage{
					Role:       "tool",
					ToolCallID: toolCall.ID,
					Name:       toolCall.Function.Name,
					Content:    res,
				})
			}

			continue
		}

		return &Generation{
			Text:       assistantMessage.Content,
			Roundtrips: i,
			Messages:   messages,
		}, nil
	}

	return nil, fmt.Errorf("exceeded completion call roundtrip limit %v", roundtripLimit)
}
