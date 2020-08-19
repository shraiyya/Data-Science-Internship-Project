import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import sys
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
import matplotlib.pyplot as plt

df=pd.read_csv(sys.argv[1])

with PdfPages('Visualization-output.pdf') as pdf:

    a=df['Areas of interest'].value_counts()
    fig1 =plt.figure(figsize = (9,4))
    plt.title('Number of students applied to different technologies',pad=90,fontsize=14)
    plt.pie(a, labels=a.index, autopct='%2.2f%%',radius=1.7,shadow=True)
    pdf.savefig()
    plt.close()

    #2 The number of students applied for Data Science who knew ‘’Python” and who didn’t
    le= LabelEncoder()
    df['areas']= le.fit_transform(df['Areas of interest'])
    bp=df.loc[(df['areas']==5) & (df['Programming Language Known other than Java (one major)']=='Python')].shape[0] #5== belong to data science
    bnp=df.loc[(df['areas']==5) & (df['Programming Language Known other than Java (one major)']!='Python')].shape[0]
    fig1 =plt.figure(figsize = (5,5))
    plt.grid(True,color='grey', linestyle='-', linewidth=0.25,alpha=0.7)
    plt.bar(['Knows Python','Does not know python'],[bp,bnp],color='pink',width = 0.35)
    plt.yticks(np.arange(0, 600, 100))
    plt.title("The number of students applied for Data Science who knew ‘’Python” and who didn’t")
    plt.xlabel("CATEGORIES")
    plt.ylabel("NUMBER OF STUDENTS")
    pdf.savefig()
    plt.close()


    #3 How Did You Hear About This Internship?
    c=df['How Did You Hear About This Internship?'].value_counts()
    fig1 =plt.figure(figsize = (9,6))
    plt.bar(c.index,c,color='coral',width = 0.55)
    plt.yticks(np.arange(0, 1300, 100))
    plt.xticks(rotation=20)
    plt.title("The different ways students learned about this program")
    plt.xlabel("MEDIUMS")
    plt.ylabel("VALUES")
    pdf.savefig()
    plt.close()


    # 4.Students who are in the fourth year and have a CGPA greater than 8.0.

    d=df[(df['Which-year are you studying in?'] =='Fourth-year') &(df['CGPA/ percentage']>= 8.0)].shape[0] 
    d1=df[(df['Which-year are you studying in?'] =='Fourth-year') &(df['CGPA/ percentage']< 8.0)].shape[0]

    explode = (0.1, 0)  
    plt.pie([d,d1], labels=['CGPA/ percentage more than 8','CGPA/ percentage less than 8'],explode=explode, autopct='%2.2f%%',radius=1,shadow=True,colors=['plum','orangered'])
    plt.title('Students who are in the fourth year and their CGPA analysis')
    pdf.savefig()
    plt.close()


    # # 5) Students who applied for Digital Marketing with verbal and written communication score greater than 8.


    e1=df[(df['areas'] ==7) &(df['Rate your written communication skills [1-10]']>= 8.0) & (df['Rate your verbal communication skills [1-10]']>= 8.0)]
    e2=df[(df['areas'] ==7) &(df['Rate your written communication skills [1-10]']< 8.0) & (df['Rate your verbal communication skills [1-10]']>= 8.0)]
    e3=df[(df['areas'] ==7) &(df['Rate your written communication skills [1-10]']>= 8.0) & (df['Rate your verbal communication skills [1-10]']< 8.0)]
    e4=df[(df['areas'] ==7) &(df['Rate your written communication skills [1-10]']< 8.0) & (df['Rate your verbal communication skills [1-10]']< 8.0)]

    e1s=e1.shape[0]
    e2s=e2.shape[0]
    e3s=e3.shape[0]
    e4s=e4.shape[0]
    
    x=['Both scores above 8','Both scores less than 8','Written communication skills score less than 8','Verbal communication skills score less 8']
    y=[e1s,e4s,e2s,e3s]

    fig1, ax = plt.subplots()
    plt.grid(True)

    ax.barh(x,y,color='lavender')
    ax.set_xticks(np.arange(0, 300, 50))
    ax.set_title("Students who applied for Digital Marketing and their verbal and written communication score  analysis",fontsize=14,pad=30)
    ax.set_ylabel("CATEGORIES")
    ax.set_xlabel("NUMBER OF STUDENTS")
                 
    pdf.savefig()
    plt.close()


    # 6 YEAR WISE CLASSIFICATION 


    f=df['Expected Graduation-year'].value_counts()
    fig1 =plt.figure(figsize = (7,7))

    plt.bar(f.index,f,color='tan',width = 0.40)
    plt.yticks(np.arange(0, 2700, 200),color='navy')
    plt.xticks(np.arange(2020,2024,1),color='navy')

    plt.title("Year-wise classification of students.",color='purple',fontsize=15)
    plt.xlabel("Years",color='purple')
    plt.ylabel("Number",color='purple')

    plt.grid(True,color='grey', linestyle='-', linewidth=0.25)
    pdf.savefig()
    plt.close()


    # 6 AREA OF STUDY CLASSIFICATION


    f1=df['Major/Area of Study'].value_counts()
    fig1 =plt.figure(figsize = (8,9))

    plt.bar(f1.index,f1,color='teal',width = 0.45)
    plt.yticks(np.arange(0, 6000, 300),color='midnightblue')
    plt.xticks(rotation=0,color='midnightblue')

    plt.title("Major-wise classification of students.",color='seagreen',fontsize=20,pad=20)
    plt.xlabel("Engineering Branches",color='seagreen',fontsize=15)
    plt.ylabel("Number of Students",color='seagreen',fontsize=15)

    plt.grid(True,color='lightgrey',linestyle='-', linewidth=1)
    pdf.savefig()
    plt.close()


    # 7 CITY WISE CLASSIFICATION
    g=df['City'].value_counts()
    fig1 =plt.figure(figsize = (9,4))
    c=['goldenrod','wheat','gold','khaki','burlywood','lemonchiffon']
    plt.title('Number of students according to city analysis',pad=60,fontsize=14)
    plt.pie(g, labels=g.index, autopct='%2.2f%%',colors=c,explode=(0.1,0,0.1,0,0.2,0),radius=1.3,shadow=True)
    pdf.savefig()
    plt.close()


    #  7 COLLEGE WISE CLASSIFICATION
    g1=df['College name'].value_counts()
    g1.sort_values(inplace=True)
    fig1 =plt.figure(figsize = (9,4))
    e=(0.1,0.1,0,0,0,0,0.1,0,0,0,0,0,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0)
    plt.title('Number of students according to colleges analysis',pad=400,fontsize=34)
    plt.pie(g1, labels=g1.index, autopct='%4.2f%%',radius=4.7,shadow=True,explode=e)
    pdf.savefig()
    plt.close()


    # 8 relationship between CGPA & Target Variable
    
    labelencoder = LabelEncoder()
    df['output'] = labelencoder.fit_transform(df['Label'])
    plt.scatter(df["CGPA/ percentage"],df["output"])
    plt.ylabel("Target Variable")
    plt.xlabel('CGPA')
    plt.title("Relationship between CGPA And Target Variable")
    plt.grid(True)
    pdf.savefig()
    plt.close()

    
    # 9 relationship between Area of interest & Target Variable

    plt.scatter(df["areas"],df["output"])
    plt.ylabel("Target Variable")
    plt.xlabel('Area of Interest')
    plt.title("Relationship between Area of Interest and Target Variable")
    plt.grid(True)
    pdf.savefig()
    plt.close()


    # 10) Plot the relationship between Year of study ,Major and Target Variable

    df['year1'] = labelencoder.fit_transform(df['Which-year are you studying in?'])
    df['major1'] = labelencoder.fit_transform(df['Major/Area of Study'])
    plt.scatter(df["major1"],df["output"])
    plt.ylabel("Target Variable")
    plt.xlabel('Major')
    plt.title("Relationship between Major of Study and Target Variable")
    plt.grid(True)
    pdf.savefig()
    plt.close()

    plt.scatter(df["year1"],df["output"])
    plt.ylabel("Target Variable")
    plt.xlabel('Year of Study')
    plt.title("Relationship between Year of Study and Target Variable")
    plt.grid(True)
    pdf.savefig()
    plt.close()

    print("PDF IS CREATED")
