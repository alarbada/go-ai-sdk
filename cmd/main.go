package main

import (
	"context"
	"encoding/json"
	"fmt"
	"go-ai-sdk"
	"log"
	"net/http"
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

var calculatorTool = ai.NewTool(
	"calculate", `
A tool for evaluating mathematical expressions.
Example expressions: '1.2 * (2 + 4.5)', '12.7 cm to inch', 'sin(45 deg) ^ 2'.
`,
	func(params struct {
		Expression string `json:"expression" jsonschema:"description=The mathematical expression to evaluate"`
	}) (any, error) {
		expr, err := govaluate.NewEvaluableExpression(params.Expression)
		if err != nil {
			return 0, err
		}

		return expr.Evaluate(nil)
	},
)

func calculatorExample() {
	ctx := context.Background()
	generator := ai.TextGenerator{
		Client: openai.NewClient(apikey),
		Model:  "gpt-4o-mini",
		System: `
You are solving math problems.
Reason step by step.
Use the calculator when necessary.
When you give the final answer,
provide an explanation for how you arrived at it.`,
		Prompt: `
A taxi driver earns $9461 per 1-hour of work.
If he works 12 hours a day and in 1 hour
he uses 12 liters of petrol with a price of $134 for 1 liter.
How much money does he earn in one day?`,
		Tools: []ai.Tool{calculatorTool},
	}

	res, err := generator.Generate(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Text)
}

func githubRepoSearcher() {
	ctx := context.Background()
	searchTool := ai.NewTool(
		"search",
		"A tool to search github repositories",
		func(params struct {
			Query string `jsonschema:"description=The search query"`
		}) (any, error) {
			url := fmt.Sprintf(
				"https://api.github.com/search/repositories?q=%v",
				params.Query,
			)
			res, err := http.Get(url)
			if err != nil {
				return nil, err
			}
			defer res.Body.Close()

			var repos struct {
				Items []struct {
					Name        string `json:"name"`
					Description string `json:"description"`
					Language    string `json:"language"`
				}
			}

			if err := json.NewDecoder(res.Body).Decode(&repos); err != nil {
				return nil, err
			}

			return repos, nil
		},
	)

	repoContributorsTool := ai.NewTool(
		"fetchRepoContributors",
		"Fetches all contributors of a repository",
		func(params struct {
			Owner string `json:"owner" jsonschema:"description=the owner of the repo"`
			Repo  string `json:"repo"  jsonschema:"description=the repository name"`
		}) (any, error) {
			url := fmt.Sprintf(
				"https://api.github.com/repos/%s/%s/contributors",
				params.Owner, params.Repo,
			)
			res, err := http.Get(url)
			if err != nil {
				return nil, err
			}
			defer res.Body.Close()

			var contributors any
			if err := json.NewDecoder(res.Body).Decode(&contributors); err != nil {
				return nil, err
			}

			return contributors, nil
		},
	)
	openai.NewClient(apikey).CreateThread(ctx, openai.ThreadRequest{
		Messages:      []openai.ThreadMessage{},
		Metadata:      map[string]any{},
		ToolResources: &openai.ToolResourcesRequest{},
	})

	generator := ai.TextGenerator{
		Client: openai.NewClient(apikey),
		Model:  "gpt-4o-mini",
		System: `
You are a github repo searcher. Reason step by step.
Use the tools available to answer questions.
`,
		Prompt: `How many contributors does langchain have?`,
		Tools:  []ai.Tool{searchTool, repoContributorsTool},
	}

	res, err := generator.Generate(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Text)
}

func main() {
	githubRepoSearcher()
}
