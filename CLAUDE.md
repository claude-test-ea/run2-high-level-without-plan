# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Always do:

At the **very start** of processing any **user message**, append the prompts to @checks/input_prompts.txt and log in the input prompt, start time and completion time in a csv called '@checks/timer.csv'
If the files do not exist then create them first. If the files pre-exist then append them.
## Project Overview

This is a loan prediction/classification project. The repository currently contains a dataset for predicting loan approval status.

## Data

- `data/input.csv` — Loan application dataset (614 rows, 13 columns)
- **Target variable:** `Loan_Status` (Y/N)
- **Features:** Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
- **Note:** The dataset contains missing values in several columns (LoanAmount, Loan_Amount_Term, Credit_History, Self_Employed, Gender, Dependents)
