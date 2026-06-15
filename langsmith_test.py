# # test_langsmith.py

# from dotenv import load_dotenv
# load_dotenv()

# import os

# print("API KEY:", bool(os.getenv("LANGSMITH_API_KEY")))
# print("TRACING:", os.getenv("LANGSMITH_TRACING"))
# print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))

# from langsmith import traceable


# @traceable(name="test_trace")
# def hello():
#     print("inside function")
#     return "hello"


# hello()