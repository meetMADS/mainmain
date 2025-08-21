import pandas as pd
import numpy as np

names = ['Manit','Naman', 'Nilesh','Karan','Eva','Nimay','Namish','Nilabha','Nilay','Nikunj']

subs = ['Math','Science','English','Math','Science','English','Math','Science','English','Math']

marks = np.random.randint(50,101, size = 10)

# making the dataframe
df = pd.DataFrame({
    'Name': names,
    'Subject': subs,
    'Score': marks,
    'Grade': ''  
})

def assign_grade(s):
    if s>=90:
        return 'A'
    elif s>=80:
        return 'B'
    elif s>=70:
        return 'C'
    elif s>=60:
        return 'D'
    else:
        return 'F'

df['Grade'] = df['Score'].apply(assign_grade)  # updating grade column

sorted_df = df.sort_values(by='Score', ascending = False)

avg = df.groupby('Subject')['Score'].mean()

# function to get those who did well
def pandas_filter_pass(data):
      good = data[data['Grade'].isin(['A','B'])]
      return good
  
pass_df = pandas_filter_pass(df)

