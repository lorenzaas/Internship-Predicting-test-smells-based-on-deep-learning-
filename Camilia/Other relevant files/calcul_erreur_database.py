import pandas as pd

# get the file

file_path = 'Result_Manual_Validation FINAL.xlsx'
data = pd.read_excel(file_path, nrows=1000)

# list of test smell columns
test_smell_columns = [
    ('Assertion Roulette', 'AssertionRoulette_Manual'),
    ('Conditional Test Logic', 'ConditionalTestLogic_Manual'),
    ('Constructor Initialization', 'ConstructorInitialization_Manual'),
    ('Duplicate Assert', 'DuplicateAssert_Manual'),
    ('Default Test', 'DefaultTest_Manual'),
    ('Dependent Test', 'DependentTest_Manual'),
    ('EmptyTest', 'EmptyTest_Manual'),
    ('Eager Test', 'EagerTest_Manual'),
    ('Exception Catching Throwing', 'ExceptionCatchingThrowing_Manual'),
    ('General Fixture', 'GeneralFixture_Manual'),
    ('IgnoredTest', 'IgnoredTest_Manual'),
    ('Magic Number Test', 'MagicNumberTest_Manual'),
    ('Mystery Guest', 'MysteryGuest_Manual'),
    ('Print Statement', 'PrintStatement_Manual'),
    ('Redundant Assertion', 'RedundantAssertion_Manual'),
    ('Resource Optimism', 'ResourceOptimism_Manual'),
    ('Sensitive Equality', 'SensitiveEquality_Manual'),
    ('Sleepy Test', 'SleepyTest_Manual'),
    ('Unknown Test', 'UnknownTest_Manual'),
    ('Verbose Test', 'VerboseTest_Manual')
]


errors = 0
total_comparisons = 0

# Iterate through each pair of JNOSE and manual test smell columns

for jnose_col, manual_col in test_smell_columns:
    # fill NaN values with 0
    jnose_data = data[jnose_col].fillna(0)
    manual_data = data[manual_col].fillna(0)

    # Count mismatches where JNOSE indicates a smell but manual validation does not, or vice versa

    mismatches = ((jnose_data > 0) & (manual_data == 0)) | ((jnose_data == 0) & (manual_data == 1))
    errors += mismatches.sum()

    # Count total comparisons made (where manual data is not NaN)
    total_comparisons += manual_data.notna().sum()

# percentage of errors
error_percentage = (errors / total_comparisons) * 100 if total_comparisons > 0 else 0
print(f"Percentage of error is: {error_percentage:.2f}%")
