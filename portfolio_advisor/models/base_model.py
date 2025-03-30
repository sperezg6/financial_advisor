import ollama

class FinancialAdvisorLLM:
    """
        A class to interact with Ollama models for financial advisory purposes.
        Uses the Ollama API to send messages and receive responses.
    """

    def __init__(self, model_name="llama3:latest"):
        """
            Initializes the FinancialAdvisorLLM with a specified model name.
        """
        self.model_name = model_name


    def ask(self, question, system_prompt=None, temperature=0.0):

        """
            Asks the financial advisor LLM for an answer to the given question.
            Returns:
                str: The response from the LLM.
        """

        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        # Add user question
        messages.append({
            'role': 'user',
            'content': question
        })

        # Call the Ollama API
        response = ollama.chat(
            self.model_name,
            messages=messages,
            options={"temperature": temperature}
        )


        return response['message']['content']
    

    def test_connection(self):
        """Test connection to the Ollama model"""
        try:
            response = ollama.chat(
                self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Hello, are you working?'
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            print(f"Ollama connection test failed: {e}")
            raise e
            
    def get_structured_response(self, question, system_prompt=None, format_instructions=None):
        """
            Get a response that follows specific formatting rules
        
            Args:
                question (str): The question to ask
                system_prompt (str, optional): System prompt for the model
                format_instructions (str, optional): Instructions on how to format the response
                
            Returns:
                str: The formatted response    
        """


        # Added formating instructions if needed
        if format_instructions:
            question = f"{question}\n\n{format_instructions}"

        return self.ask(question, system_prompt=system_prompt, temperature=0.0)
    
    