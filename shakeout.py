import ollama
response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'You are a Site Reliability Engineer. Give me a one-sentence summary of your job.'}])
print(response['message']['content'])