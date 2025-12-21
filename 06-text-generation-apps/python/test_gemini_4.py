import google.generativeai as genai
import os
import re
from dotenv import load_dotenv

load_dotenv()


gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.configure(api_key=gemini_api_key)



if not gemini_api_key:
    print("Error: GEMINI_API_KEY not found.")
    exit()




def get_gemini_response(sys_ins,prompt,temperature):
    try:
        model = genai.GenerativeModel(
            "models/gemini-2.0-flash",
            system_instruction=sys_ins,
            generation_config={
                "temperature": temperature,  # Vary this to control randomness
                "max_output_tokens": 1000  # Adjust output length if needed
                }
                )
        response = model.generate_content(prompt)
        print(response.text)
        return response.text  # Return the text part of the response
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return None


def parse_quiz_content(quiz_content):
    # Regex patterns to extract the questions, options, and correct answers
    question_pattern = r'\*(\d+\..*?)\*'  # Capture the question text within *...*
    option_pattern = r'\*\*([A-D])\. (.*?)\*\*'  # Capture the options in **A. ...**
    # Update the answer pattern to capture correct answers and their explanations
    answer_pattern = r'\*\*\*([A-D]) is the correct answer!.*?\*\*\*'  # Capture the answer like ***B is the correct answer!***

    # Extract the questions
    questions = re.findall(question_pattern, quiz_content)
    # Extract the options (grouped by question)
    options = re.findall(option_pattern, quiz_content)
    # Extract the correct answers (just the answer letters)
    answers = re.findall(answer_pattern, quiz_content)

    return questions, options, answers


def start_quiz():
    print("Welcome to the interactive quiz! Let's get started.")

    # Step 1: Get subject and grade level from the user
    subject = input("What subject would you like to be quizzed on? ")
    grade_level = input("What is your grade level? (e.g., 10) ")

    # Step 2: Generate quiz prompt for Gemini API
    sys_ins="""
    I want you to act as a friendly teacher who knows the subject of {subject} at my grade level of {grade_level}."""
    quiz_prompt = f"""
    I want you to act as a friendly teacher who knows the subject of {subject} at my grade level of {grade_level}.
    Please prepare a multiple choice quiz with 10 questions. Each question should have four options (A, B, C, D).
    Please prepare EXACTLY 10 multiple-choice questions. Each question should have four options (A, B, C, D). Make sure to include the correct answer and provide a brief explanation after each question.
    Guidelines:
        1. Each time, generate a fresh set of questions. **Do not repeat previously generated questions.**
        2. Ensure questions vary in difficulty—mix easy, medium, and challenging questions.
        3. Include both conceptual and trivia-based questions for a balanced quiz.
        4. Questions must be suitable for the specified grade level.
        5. Use different formats (e.g., factual, scenario-based, historical references, and "which of the following" questions).
        6. Mark questions with * at the start and end.
        7. Mark options with ** at the start and end.
        8. Mark the correct answer with *** at the start and end.

    For example *1. What do we use to hit the ball in cricket?*

**A. A bat**
**B. A stick**
**C. Our hands**
**D. A racquet**

***A is the correct answer!*** We use a bat to hit the ball in cricket!


*2. What shape is a cricket ball?*

**A. Square**
**B. Triangle**
**C. Round**
**D. Rectangle**

***C is the correct answer!***  A cricket ball is round, just like a bouncy ball! and so on.. DO NOT generate fewer than 10 questions. If you need more space, summarize explanations briefly.Generate **new and unique** questions every time..
    """

    # Step 3: Get quiz questions from Gemini
    quiz_content = get_gemini_response(sys_ins,quiz_prompt,0.9)
    print(quiz_content)
    if not quiz_content:
        print("Failed to get quiz. Please try again later.")
        return
    print(quiz_content)
    # Step 4: Parse the quiz content into questions, options, and answers
    questions, options, answers = parse_quiz_content(quiz_content)
    print("list of questions are: ")
    print(questions)
    print("list of options are: ")
    print(options)
    print("list of answers are: ")
    print(answers)

    # Display parsed content
    print("\nGreat! Let's start the quiz.")
    correct_answers = 0
    total_questions = 10

    for i in range(total_questions):
        print(f"\nQuestion {i + 1}: {questions[i]}")

        # Ensure the options are correctly displayed
        option_dict = {option[0]: option[1] for option in options[i*4:(i+1)*4]}  # Group options by question
        for key, value in option_dict.items():
            print(f"{key}. {value}")

        # Get the user's answer
        user_answer = input("Please select A, B, C, or D: ").strip().upper()
        print("test answers")
        print(answers[i])
        print(answers[i][0])

        # Check if the answer is valid
        if user_answer in ['A', 'B', 'C', 'D']:
            
            # Check if user's answer matches the correct answer
            correct_answer = answers[i][0]
            if user_answer == correct_answer:
                correct_answers += 1
                print(f"Correct! The answer is {correct_answer}.")
            else:
                print(f"Oops! The correct answer is {correct_answer}.")
        else:
            print("Invalid answer, please select A, B, C, or D.")

    # Step 5: End the quiz and provide feedback
    print(f"\nQuiz complete! You answered {correct_answers}/{total_questions} questions correctly.")
    print("Thanks for participating! I hope you learned something new today.")

# Run the quiz
start_quiz()


