package main

import (
	"context"
	"fmt"
	"go-ai-sdk"
	"os"
	"sync"

	"github.com/Knetic/govaluate"
	"github.com/gookit/goutil/dump"
	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
)

var loadenv = sync.OnceFunc(func() {
	godotenv.Load()
})

func getenv(key string) string {
	loadenv()
	return os.Getenv(key)
}

func init() {
	dump.Config(func(opts *dump.Options) {
		opts.MaxDepth = 10
		opts.ShowFlag = 0
	})
}

var apikey = getenv("OPENAI_API_KEY")

type calculateParams struct {
	Expression string `json:"expression" jsonschema:"description=The mathematical expression to evaluate"`
}

func main() {
	ctx := context.Background()
	generator := ai.TextGenerator{
		Client: openai.NewClient(apikey),
		Model:  "gpt-4o-mini",
		System: "You are solving math problems. " +
			"Reason step by step. " +
			"Use the calculator when necessary. " +
			"When you give the final answer, " +
			"provide an explanation for how you arrived at it.",
		Prompt: "A taxi driver earns $9461 per 1-hour of work. " +
			"If he works 12 hours a day and in 1 hour " +
			"he uses 12 liters of petrol with a price of $134 for 1 liter. " +
			"How much money does he earn in one day?",
		Tools: ai.Tools{
			"calculate": ai.NewTool(
				"A tool for evaluating mathematical expressions. "+
					"Example expressions: "+
					"'1.2 * (2 + 4.5)', '12.7 cm to inch', 'sin(45 deg) ^ 2'.",
				func(params calculateParams) (any, error) {
					expr, err := govaluate.NewEvaluableExpression(params.Expression)
					if err != nil {
						return 0, err
					}

					return expr.Evaluate(nil)
				},
			),
		},
	}

	res, err := generator.Generate(ctx)
	fmt.Println(res, err)
}
