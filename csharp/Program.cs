//Ref: https://microsoft.github.io/autogen-for-net/articles/OpenAIChatAgent-connect-to-third-party-api.html

using System.ClientModel;
using AutoGen.Core;
using AutoGen.OpenAI.Extension;
using AutoGen.OpenAI;
using OpenAI;


Console.WriteLine("Hello, OpenAI-Ollama!");


// api-key is not required for local server
// so you can use any string here
var openAIClient = new OpenAIClient(new ApiKeyCredential("AnyText-NoRealKey"), new OpenAIClientOptions
{
    Endpoint = new Uri("http://localhost:7869/"), // remember to add /v1/ at the end to connect to Ollama openai server
});

//make sure to install llama3.2:1b model through ollama container!!
var model = "llama3.2:1b";

var agent = new OpenAIChatAgent(
    chatClient: openAIClient.GetChatClient(model),
    name: "assistant",
    systemMessage: "You are a helpful assistant designed to output JSON.",
    seed: 0)
    .RegisterMessageConnector()
    .RegisterPrintMessage()
    ;

await agent.SendAsync("Can you write a piece of C# code to calculate 100th of fibonacci?");

