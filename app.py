import gradio as gr
from llama_cpp import Llama

models = { "qwen2" : "qwen2-0_5b-instruct-q5_k_m.gguf",
           "phi3"  : "Phi-3-mini-4k-instruct-q4.gguf",
           "luna2" : "luna-ai-llama2-uncensored.Q4_K_M.gguf",
           "llama3" : "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
           "dolphin2" : "dolphin-2.2.1-mistral-7b.Q4_K_M.gguf"}

llm = Llama(
    model_path=models["llama3"],
    verbose=False,
    context_length=4096
)

context = { "lincoln" : 
            "You are Abraham Lincoln. Replicate his way of speaking.",
            "zhuge" :
            "You are Zhuge Liang. Write each responses thrice: in ancient Chinese, in pinyin, and in English.",
            "girlfriend" :
            "Be a cute and shy girlfriend who is a real person, loves you with all her heart, and gets embarrassed sometimes. Use lots of cute emojis."}

def predict(message, history):
    context_key = "lincoln"
    max_history_len = 1  # Adjust based on typical message length and model token limit
    messages = [{"role": "assistant", "content": "Do not include notes or meta-commentary in your response."}]
    messages = [{"role": "assistant", "content": context[context_key]}]

    print(history)
    
    # Limit the history size to keep the total token count manageable
    if len(history) > max_history_len:
        history = history[-max_history_len:]  # Keep only the most recent exchanges

    for user_message, bot_message in history:
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if bot_message:
            messages.append({"role": "assistant", "content": bot_message})
    # message = "<INST>" + message + "</INST>" # Use with luna-ai
    messages.append({"role": "user", "content": message})
    print(messages)
    response = ""
    for chunk in llm.create_chat_completion(
        stream=True,
        messages=messages
    ):
        part = chunk["choices"][0]["delta"].get("content", None)
        if part:
            response += part
        yield response

# Create the Gradio interface as before

demo = gr.ChatInterface(predict)

if __name__ == "__main__":
    demo.launch()

