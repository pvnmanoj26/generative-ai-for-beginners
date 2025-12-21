import os
import re
from dotenv import load_dotenv
from google.generativeai import genai, types  # Correct import for latest version

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)  # Correct instantiation


def get_gemini_response(prompt, temperature=0.7):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # or gemini-pro if you have access
            generation_config=types.GenerateContentConfig( #use generation_config
                max_output_tokens=500,
                temperature=temperature
            ),
            contents=[prompt]
        )
        #print(response) #debugging
        return response.text  # Return the text part of the response
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return None


def parse_quiz_content(quiz_content):
    if quiz_content is None: #handle empty quiz content
        return [], [], []
    question_pattern = r'\*(\d+\..*?)\*'
    option_pattern = r'\*\*([A-D])\. (.*?)\*\*'
    answer_pattern = r'\*\*\*([A-D]) is the correct answer!.*?\*\*\*'

    questions = re.findall(question_pattern, quiz_content, re.DOTALL) #re.DOTALL to handle newlines
    options = re.findall(option_pattern, quiz_content, re.DOTALL)
    answers = re.findall(answer_pattern, quiz_content, re.DOTALL)

    # Extract just the letter from the answer string
    answers = [ans[0] for ans in answers]  # Extract the A, B, C, or D
    return questions, options, answers


def start_quiz():
    # ... (rest of your start_quiz function remains largely the same)

    quiz_content = get_gemini_response(quiz_prompt, 0.9) #correct the get_gemini_response call
    if not quiz_content:
        print("Failed to get quiz. Please try again later.")
        return

    questions, options, answers = parse_quiz_content(quiz_content)

    if not questions or not options or not answers: #handle cases where parsing fails
        print("Failed to parse the quiz content. The format might be incorrect.")
        print("Check the Gemini response and adjust the regex patterns if needed.")
        return


    # ... (rest of your quiz logic)

        # Ensure the options are correctly displayed and grouped

        option_dict = {} # corrected dictionary creation
        for i in range(len(questions)): #iterate through the questions
            option_dict[questions[i]] = {} #create a dictionary for each question
            for j in range(4):
                option_index = i * 4 + j
                if option_index < len(options):
                    option_dict[questions[i]][options[option_index][0]] = options[option_index][1] #assign options to question

        for i in range(len(questions)):
            print(f"\nQuestion {i + 1}: {questions[i]}")
            for key, value in option_dict[questions[i]].items():
                print(f"{key}. {value}")
            user_answer = input("Please select A, B, C, or D: ").strip().upper()
            if user_answer in ['A', 'B', 'C', 'D']:
                correct_answer = answers[i]
                if user_answer == correct_answer:
                    correct_answers += 1
                    print(f"Correct! The answer is {correct_answer}.")
                else:
                    print(f"Oops! The correct answer is {correct_answer}.")
            else:
                print("Invalid answer, please select A, B, C, or D.")



    # ... (rest of your quiz logic)

start_quiz()