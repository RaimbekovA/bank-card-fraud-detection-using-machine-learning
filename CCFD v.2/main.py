# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import  confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler, PowerTransformer

import streamlit as st

import warnings
warnings.filterwarnings("ignore")



# Title
st.title('Bank Card Fraud Detection')


# Read Data into a Dataframe
df = pd.read_csv('creditcard.csv')


# --- 1 CHECKBOX ---
# Print description of the initial data and shape
if st.sidebar.checkbox('Show the initial data set'):
    st.header("Understanding dataset")

    st.write('Initial data set: \n', df)
    st.write('Data decription: \n', df.describe())
    st.write('Shape of the dataframe: ',df.shape)
    st.text('The dataset consists of 284,807 rows and 31 columns.\nThere is no zero value in the data.')

    st.header("Checking missing and outlier values")

    # Check missing values
    st.write('Missing values: ', df.isnull().values.sum())

    # Checking the number of missing values in each column
    st.write('The number of missing values in each column: ', df.isnull().sum())

    # Percentage of null values
    percent_missing = (df.isnull().sum().sort_values(ascending = False) / len(df)) * 100
    st.write('Percentage of null values: ', percent_missing)

    # Check if there are any duplicate rows
    st.write('Duplicate rows: ', df.duplicated(keep=False).sum())

    # Delete duplicate rows
    df = df.drop_duplicates() 
    st.write('Deleting duplicate rows was successful. This is a new data set:', df)
# --- 1 CHECKBOX ---



# --- 2 CHECKBOX ---
if st.sidebar.checkbox('Show the analysis'):
    
    fraud = df[df.Class == 1]
    valid = df[df.Class == 0]

    outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100

    st.header('Univariate analysis')

    st.write('Fraud Cases: ', len(fraud))
    st.write('Valid Cases: ', len(valid))
    st.write('Compare the values for both transactions: \n', df.groupby('Class').mean())
    st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)


    # Method to compute countplot of given dataframe parameters:
    # - data(pd.Dataframe): Input Dataframe
    # - feature(str): Feature in Dataframe
    def countplot_data(data, feature):
        plt.figure(figsize=(10,10))
        sns.countplot(x=feature, data=data)
        plt.show()

    # Method to construct pairplot of the given feature wrt data parameters:
    # - data(pd.DataFrame): Input Dataframe
    # - feature1(str): First Feature for Pair Plot
    # - feature2(str): Second Feature for Pair Plot
    # - target: Target or Label (y)
    def pairplot_data_grid(data, feature1, feature2, target):
        sns.FacetGrid(data, hue=target).map(plt.scatter, feature1, feature2).add_legend()
        plt.show()

    st.subheader('Transaction ratio:')
    st.pyplot(countplot_data(df, df.Class))

    st.subheader('The relationship of fraudulent transactions with the amount of money:\n')
    st.pyplot(pairplot_data_grid(df, "Time", "Amount", "Class"))
    


    st.header('Bivariate Analysis')
    
    st.write('Fraud: ', df.Time[df.Class == 1].describe())
    st.write('Not fraud: ', df.Time[df.Class == 0].describe())

    
    def graph1():
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
        bins = 50

        ax1.hist(df.Time[df.Class == 1], bins = bins)
        ax1.set_title('Fraud')

        ax2.hist(df.Time[df.Class == 0], bins = bins)
        ax2.set_title('Not Fraud')

        plt.xlabel('Time (Sec.)')
        plt.ylabel('Number of Transactions')
        plt.show()


    def graph2():
        f, axes = plt.subplots(ncols=2, figsize=(16,10))
        colors = ['#C35617', '#FFDEAD']

        sns.boxplot(x="Class", y="Amount", data=df, palette = colors, ax=axes[0], showfliers=True)
        axes[0].set_title('Class vs Amount')

        sns.boxplot(x="Class", y="Amount", data=df, palette = colors, ax=axes[1], showfliers=False)
        axes[1].set_title('Class vs Amount without outliers')

        plt.show()

    
    def graph3():
        fig, ax = plt.subplots(1, 2, figsize=(18,4))

        amount_val = df['Amount'].values
        time_val = df['Time'].values

        sns.distplot(amount_val, ax=ax[0], color='b')
        ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
        ax[0].set_xlim([min(amount_val), max(amount_val)])

        sns.distplot(time_val, ax=ax[1], color='r')
        ax[1].set_title('Distribution of Transaction Time', fontsize=14)
        ax[1].set_xlim([min(time_val), max(time_val)])

        plt.show()


    st.pyplot(graph1())
    st.pyplot(graph2())
    st.pyplot(graph3())



    st.header('Multivariate Analysis')


    # Plot relation with different scale
    def graph4(): 
        df1 = df[df['Class']==1]
        df2 = df[df['Class']==0]
        fig, ax = plt.subplots(1,2, figsize=(15, 5))

        ax[0].scatter(df1['Time'], df1['Amount'], color='red', marker= '*', label='Fraudrent')
        ax[0].set_title('Time vs Amount')
        ax[0].legend(bbox_to_anchor =(0.25, 1.15))

        ax[1].scatter(df2['Time'], df2['Amount'], color='green', marker= '.', label='Non Fraudrent')
        ax[1].set_title('Time vs Amount')
        ax[1].legend(bbox_to_anchor =(0.3, 1.15))

        plt.show()


    def graph5():
        sns.lmplot(x='Time', y='Amount', hue='Class', markers=['x', 'o'], data=df, height=6)
    

    # plot relation in same scale
    def graph6():
        g = sns.FacetGrid(df, col="Class", height=6)
        g.map(sns.scatterplot, "Time", "Amount", alpha=.7)
        g.add_legend()
    

    st.pyplot(graph4())
    st.pyplot(graph5())
    st.pyplot(graph6())  
# --- 2 CHECKBOX ---


# --- 3 CHECKBOX ---
if st.sidebar.checkbox('Model building on imbalanced data'):
    # --- TRAIN AND TEST SPLIT ---
    st.header('Train and test split')


    # Putting feature variables into X
    X = df.drop(['Class'], axis=1)

    # Putting target variable to y
    y = df['Class']


    # Splitting data into train and test set 80:20
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 42)

    st.write('X_train: ', X_train.shape)
    st.write('y_train: ', y_train.shape)
    st.write('X_test: ', X_test.shape)
    st.write('y_test: ', y_test.shape)
    # --- TRAIN AND TEST SPLIT ---


    
    # --- FEATURE SCALING ---
    # Instantiate the Scaler
    scaler = StandardScaler()

    # Fit the data into scaler and transform
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])


    # Transform the test set
    X_test['Amount'] = scaler.transform(X_test[['Amount']])


    # Checking the Skewness
    # Listing the columns
    cols = X_train.columns
    
    
    # Plotting the distribution of the variables (skewness) of all the columns
    def skewness(): 
        k = 0
        plt.figure(figsize=(17,28))
        for col in cols :    
            k = k + 1
            plt.subplot(6, 5,k)    
            sns.distplot(X_train[col])
            plt.title(col+' '+str(X_train[col].skew()))
    

    st.header('Checking the Skewness')
    st.pyplot(skewness())
    # --- FEATURE SCALING ---



    # --- Mitigate skwenes with PowerTransformer ---
    # Instantiate the powertransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)

    # Fit and transform the PT on training data
    X_train[cols] = pt.fit_transform(X_train)

    # Transform the test set
    X_test[cols] = pt.transform(X_test)

    
    def newSkewness():
        k=0
        plt.figure(figsize=(17,28))
        for col in cols :    
            k=k+1
            plt.subplot(6, 5,k)    
            sns.distplot(X_train[col])
            plt.title(col+' '+str(X_train[col].skew()))
    

    st.header('Mitigate skwenes with PowerTransformer')
    st.pyplot(newSkewness())
    # --- Mitigate skwenes with PowerTransformer ---   
# --- 3 CHECKBOX ---



# --- 4 CHECKBOX ---
if st.sidebar.checkbox('Compare algorithms'):
    # --- TRAIN AND TEST SPLIT ---
    # Putting feature variables into X
    X = df.drop(['Class'], axis=1)

    # Putting target variable to y
    y = df['Class']


    # Splitting data into train and test set 80:20
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 42)
    # --- TRAIN AND TEST SPLIT ---


    # --- FEATURE SCALING ---
    # Instantiate the Scaler
    scaler = StandardScaler()


    # Fit the data into scaler and transform
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])


    # Transform the test set
    X_test['Amount'] = scaler.transform(X_test[['Amount']])


    # Checking the Skewness
    # Listing the columns
    cols = X_train.columns
    # --- FEATURE SCALING ---


    # --- Mitigate skwenes with PowerTransformer ---
    # Instantiate the powertransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)

    # Fit and transform the PT on training data
    X_train[cols] = pt.fit_transform(X_train)

    # Transform the test set
    X_test[cols] = pt.transform(X_test)
    # --- Mitigate skwenes with PowerTransformer ---


    def visualize_confusion_matrix(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Oranges',
                    xticklabels=['No Credit Card Fraud Dection','Credit Card Fraud Dection'], 
                    yticklabels=['No Credit Card Fraud Dection','Credit Card Fraud Dection'])
        plt.title('Accuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))
        plt.ylabel('True Values')
        plt.xlabel('Predicted Values')
        plt.show()
        
        st.write("\n")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        return

    
    def ROC_AUC(Y, Y_prob):
        # caculate roc curves
        fpr, tpr, threshold = roc_curve(Y, Y_prob)
        # caculate scores
        model_auc = roc_auc_score(Y, Y_prob)
        # plot roc curve for the model
        plt.figure(figsize=(16, 9))
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label='Model - AUC=%.3f' % (model_auc))
        # show axis labels and the legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show(block=False)
        return


    # --- START Logistic regression ---
    st.header('Logistic Regression')
    # --- START Training the Logistic Regression Model on the Training set ---
    st.subheader('Training the Logistic Regression Model on the Training set')
    

    LR_model = LogisticRegression(random_state = 0)
    LR_model.fit(X_train, y_train)
    y_train_pred = LR_model.predict(X_train)
    y_test_pred = LR_model.predict(X_test)
    acc1 = accuracy_score(y_test, y_test_pred)


    # Train Score
    st.write('Recall score: %0.4f'% recall_score(y_train, y_train_pred))
    st.write('Precision score: %0.4f'% precision_score(y_train, y_train_pred))
    st.write('F1-Score: %0.4f'% f1_score(y_train, y_train_pred))
    st.write('Accuracy score: %0.4f'% accuracy_score(y_train, y_train_pred))
    st.write('AUC: %0.4f' % roc_auc_score(y_train, y_train_pred))

    # Train Predictions
    st.pyplot(visualize_confusion_matrix(y_train, y_train_pred))


    st.pyplot(ROC_AUC(y_train, y_train_pred))
    # --- END Training the Logistic Regression Model on the Training set ---


    # --- START Training the Logistic Regression Model on the Testing set ---
    st.subheader('Training the Logistic Regression Model on the Testing set')


    # Test score
    st.write('Recall score: %0.4f'% recall_score(y_test, y_test_pred))
    st.write('Precision score: %0.4f'% precision_score(y_test, y_test_pred))
    st.write('F1-Score: %0.4f'% f1_score(y_test, y_test_pred))
    st.write('Accuracy score: %0.4f'% accuracy_score(y_test, y_test_pred))
    st.write('AUC: %0.4f' % roc_auc_score(y_test, y_test_pred))


    # Test Predictions
    st.pyplot(visualize_confusion_matrix(y_test, y_test_pred))


    st.pyplot(ROC_AUC(y_test, y_test_pred))
    # --- END Training the Logistic Regression Model on the Testing set ---


    # Result
    st.header('Results')
    st.subheader('Training set')
    st.text('- Recall score: 0.6397\n- Precision score: 0.8688\n- F1-Score: 0.7368\n- Accuracy score: 0.9992\n- AUC: 0.8198')
    

    st.subheader('Testing set')
    st.text('- Recall score: 0.5556\n- Precision score: 0.9091\n- F1-Score: 0.6897\n- Accuracy score: 0.9992\n- AUC: 0.7777')
    # --- END Logistic regression ---



    # --- START Naive Bayes ---
    st.header('Naive Bayes')

    
    # --- START Training the Naive Bayes Model on the Training set ---
    st.subheader('Training the Naive Bayes Model on the Training set')


    NB_model = GaussianNB()
    NB_model.fit(X_train, y_train)
    y_train_pred = NB_model.predict(X_train)
    y_test_pred = NB_model.predict(X_test)
    acc2 = accuracy_score(y_test, y_test_pred)


    # Train Score
    st.write('Recall score: %0.4f'% recall_score(y_train, y_train_pred))
    st.write('Precision score: %0.4f'% precision_score(y_train, y_train_pred))
    st.write('F1-Score: %0.4f'% f1_score(y_train, y_train_pred))
    st.write('Accuracy score: %0.4f'% accuracy_score(y_train, y_train_pred))
    st.write('AUC: %0.4f' % roc_auc_score(y_train, y_train_pred))


    # Train Predictions
    st.pyplot(visualize_confusion_matrix(y_train, y_train_pred))


    st.pyplot(ROC_AUC(y_train, y_train_pred))
    # --- END Training the Naive Bayes Model on the Training set ---


    # --- START Training the Naive Bayes Model on the Testing set ---
    st.subheader('Training the Naive Bayes Model on the Testing set')


    # Test score
    st.write('Recall score: %0.4f'% recall_score(y_test, y_test_pred))
    st.write('Precision score: %0.4f'% precision_score(y_test, y_test_pred))
    st.write('F1-Score: %0.4f'% f1_score(y_test, y_test_pred))
    st.write('Accuracy score: %0.4f'% accuracy_score(y_test, y_test_pred))
    st.write('AUC: %0.4f' % roc_auc_score(y_test, y_test_pred))


    # Test Predictions
    st.pyplot(visualize_confusion_matrix(y_test, y_test_pred))


    st.pyplot(ROC_AUC(y_test, y_test_pred))
    # --- END Training the Naive Bayes Model on the Testing set ---


    # Result
    st.header('Results')
    st.subheader('Training set')
    st.text('- Recall score: 0.8277\n- Precision score: 0.0604\n- F1-Score: 0.1125\n- Accuracy score: 0.9780\n- AUC: 0.9030')
    

    st.subheader('Testing set')
    st.text('- Recall score: 0.7778\n- Precision score: 0.0523\n- F1-Score: 0.0980\n- Accuracy score: 0.9773\n- AUC: 0.8777')
    # --- END Naive Bayes ---

    
    
    # --- START Decision tree ---
    st.header('Decision tree')

    
    # --- START Training the Decision tree Model on the Training set ---
    st.subheader('Training the Decision tree Model on the Training set')


    DTR_model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    DTR_model.fit(X_train, y_train)
    y_train_pred = DTR_model.predict(X_train)
    y_test_pred = DTR_model.predict(X_test)
    acc4 = accuracy_score(y_test, y_test_pred)


    # Train Score
    st.write('Recall score: %0.4f'% recall_score(y_train, y_train_pred))
    st.write('Precision score: %0.4f'% precision_score(y_train, y_train_pred))
    st.write('F1-Score: %0.4f'% f1_score(y_train, y_train_pred))
    st.write('Accuracy score: %0.4f'% accuracy_score(y_train, y_train_pred))
    st.write('AUC: %0.4f' % roc_auc_score(y_train, y_train_pred))


    st.pyplot(visualize_confusion_matrix(y_train, y_train_pred))

    st.pyplot(ROC_AUC(y_train, y_train_pred))
    # --- END Training the Decision tree Model on the Training set ---


    # --- START Training the Decision tree Model on the Testing set ---
    st.subheader('Training the Decision tree Model on the Testing set')


    st.write('Recall score: %0.4f'% recall_score(y_test, y_test_pred))
    st.write('Precision score: %0.4f'% precision_score(y_test, y_test_pred))
    st.write('F1-Score: %0.4f'% f1_score(y_test, y_test_pred))
    st.write('Accuracy score: %0.4f'% accuracy_score(y_test, y_test_pred))
    st.write('AUC: %0.4f' % roc_auc_score(y_test, y_test_pred))


    st.pyplot(visualize_confusion_matrix(y_test, y_test_pred))

    st.pyplot(ROC_AUC(y_test, y_test_pred))
    # --- END Training the Decision tree Model on the Testing set ---


    # Result
    st.header('Results')
    st.subheader('Training set')
    st.text('- Recall score: 1.0000\n- Precision score: 1.0000\n- F1-Score: 1.0000\n- Accuracy score: 1.0000\n- AUC: 1.0000')
    

    st.subheader('Testing set')
    st.text('- Recall score: 0.6889\n- Precision score: 0.7561\n- F1-Score: 0.7209\n- Accuracy score: 0.9992\n- AUC: 0.8443')
    # --- END Decision tree ---



    st.header('Compare the accuracy of the models on the Testing set')

    def compareResult():
        mylist=[]
        mylist2=[]

        mylist.append(acc1)
        mylist2.append("Logistic Regression")

        mylist.append(acc2)
        mylist2.append("Naive Bayes")

        mylist.append(acc4)
        mylist2.append("Decision Tree")


        plt.figure(figsize=(22, 10))
        sns.set_style("darkgrid")
        ax = sns.barplot(x = mylist2, y = mylist, palette = "Oranges", saturation =1.5)
        plt.xlabel("Classification Models", fontsize = 20 )
        plt.ylabel("Accuracy", fontsize = 20)
        plt.title("Accuracy of different Classification Models", fontsize = 20)
        plt.xticks(fontsize = 11, horizontalalignment = 'center', rotation = 0)
        plt.yticks(fontsize = 13)
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
            
        plt.show()

    
    st.pyplot(compareResult())



    st.header('ROC Curve and Area Under the Curve')


    # Logistic Regression
    y_pred_logistic = LR_model.predict_proba(X_test)[:,1]
    logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_logistic)
    auc_logistic = auc(logistic_fpr, logistic_tpr)


    # Naive Bayes
    y_pred_nb = NB_model.predict_proba(X_test)[:,1]
    nb_fpr, nb_tpr, threshold = roc_curve(y_test, y_pred_nb)
    auc_nb = auc(nb_fpr, nb_tpr)


    # Decision Tree
    y_pred_dtr = DTR_model.predict_proba(X_test)[:,1]
    dtr_fpr, dtr_tpr, threshold = roc_curve(y_test, y_pred_dtr)
    auc_dtr = auc(dtr_fpr, dtr_tpr)


    def plottingGraphResultCompare():
        plt.figure(figsize=(10, 8), dpi=100)
        plt.plot([0, 1], [0, 1], 'k--')
        # Logistic Regression
        plt.plot(logistic_fpr, logistic_tpr, label='Logistic Regression (auc = %0.4f)' % auc_logistic)
        # Naive Bayes
        plt.plot(nb_fpr, nb_tpr, label='Naive Bayes (auc = %0.4f)' % auc_nb)

        # Decision Tree
        plt.plot(dtr_fpr, dtr_tpr, label='Decision Tree (auc = %0.4f)' % auc_dtr)


        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend(loc='best')
        plt.show()
    

    st.pyplot(plottingGraphResultCompare())

# --- 4 CHECKBOX ---



# --- 5 CHECKBOX ---
if st.sidebar.checkbox('Manual transaction verification'):

    # separate legitimate and fraudulent transactions
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]

    # undersample legitimate transactions to balance the classes
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)

    # split data into training and testing sets
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    # train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # evaluate model performance
    train_acc = accuracy_score(model.predict(X_train), y_train)
    test_acc = accuracy_score(model.predict(X_test), y_test)

    # create Streamlit app
    st.title("Manual transaction verification")
    st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

    # create input fields for user to enter feature values
    input_df = st.text_input('Input All features')
    input_df_lst = input_df.split(',')
    # create a button to submit input and get prediction
    submit = st.button("Submit")

    if submit:
        # get input feature values
        features = np.array(input_df_lst, dtype=np.float64)
        # make prediction
        prediction = model.predict(features.reshape(1,-1))
        # display result
        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")
# --- 5 CHECKBOX ---
