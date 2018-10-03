import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor


##############################################################
def readFile(file_path=r''):
    train_df = pd.read_csv(file_path + 'train.csv')
    test_df = pd.read_csv(file_path + 'test.csv')

    df = pd.concat([train_df, test_df], axis=0)
    df.index = range(len(df))
    train_idx = list(range(len(train_df)))
    test_idx = list(range(len(train_df), len(df)))

    return df, train_idx, test_idx


##############################################################
def processCabin(df):
    df['Has_Cabin'] = df['Cabin'].isnull().astype(float)
    df['Cabin'].fillna('U0', inplace=True)
    df['Cabin'] = df['Cabin'].map(lambda x: x.split(' '))
    df['Cabin_num'] = df['Cabin'].map(lambda x: len(x))
    df['Cabin_letter'] = df['Cabin'].map(lambda x: [re.compile('([a-zA-Z]+)').search(val).group() for val in x])
    Cabin_letter_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,
                         'F': 6, 'G': 7, 'T': 8, 'U': 9}
    df['Cabin_letter'] = df['Cabin_letter'].map(lambda x: np.mean([Cabin_letter_dict[val] for val in x]))
    return df


##############################################################
def processEmbarked(df):
    df['Embarked'].fillna('S', inplace=True)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df
##############################################################

def processFare(df, train_idx, test_idx):
    Fare_mean = df.loc[test_idx, 'Fare'].mean()
    df.loc[test_idx, 'Fare'] = df.loc[test_idx, 'Fare'].fillna(Fare_mean)
    return df

##############################################################

def processName(df):
    def get_title(name):
        title_search = re.search('([a-zA-Z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ''
    df['Name_title'] = df['Name'].apply(get_title)

    replace_lst = ['Lady', 'Countess','Capt', 'Col', 'Don',
            'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Name_title'] = df['Name_title'].replace(replace_lst, 'Rare')
    df['Name_title'] = df['Name_title'].replace('Mlle', 'Miss')
    df['Name_title'] = df['Name_title'].replace('Ms', 'Miss')
    df['Name_title'] = df['Name_title'].replace('Mme', 'Mrs')

    title_dict = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df['Name_title'] = df['Name_title'].map(title_dict)
    df['Name_title'].fillna(0, inplace=True)

    df['Name_length'] = df['Name'].apply(len)
    df['Name_num'] = df['Name'].map(lambda x: len(re.split(' ', x)))

    return df


##############################################################
def processSex(df):
    df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)

    return df


##############################################################
def processTicket(df):
    def getTicketPrefix(ticket):
        match = re.compile('([a-zA-Z./]+)').search(ticket)
        if match:
            return match.group()
        else:
            return 'U'

    df['Ticket_prefix'] = df['Ticket'].map(lambda x: getTicketPrefix(x.upper()))
    df['Ticket_prefix'] = df['Ticket_prefix'].map(lambda x: re.sub('[.?/?]', '', x))
    df['Ticket_prefix'] = df['Ticket_prefix'].map(lambda x: re.sub('STON', 'SOTON', x))

    df['Ticket_prefix_id'] = pd.factorize(df['Ticket_prefix'])[0]

    return df


##############################################################
def processAge(df):
    notnull_age = df.loc[df['Age'].notnull(),
            ['Age', 'Embarked', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Cabin_letter']]

    isnull_age = df.loc[df['Age'].isnull(),
            ['Age', 'Embarked', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Cabin_letter']]

    notnull_x = notnull_age.values[:, 1::]
    notnull_y = notnull_age.values[:, 0]
    clf = RandomForestRegressor(n_estimators=2000)
    clf.fit(notnull_x, notnull_y)

    pred_age = clf.predict(isnull_age.values[:, 1::])
    df.loc[df['Age'].isnull(), 'Age'] = pred_age

    return df


##############################################################
def coreProcess():
    df, train_idx, test_idx = readFile()
    df = processCabin(df)
    df = processEmbarked(df)
    df = processFare(df, train_idx, test_idx)
    df = processName(df)
    df = processSex(df)
    df = processTicket(df)
    df = processAge(df)

    return df, train_idx, test_idx
