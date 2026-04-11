import sys
import platform

print("Python Version:", sys.version)
print("Platform:", platform.platform())

libraries = [
    "numpy",
    "pandas",
    "sklearn",
    "joblib",
    "matplotlib",
    "seaborn",
    "scipy",
    "streamlit",
    "xgboost",
    "lightgbm",
    "pickle"
]

for lib in libraries:
    try:
        module = __import__(lib)
        print(f"{lib} version:", module.__version__)
    except:
        print(f"{lib} not installed")