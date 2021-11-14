import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pickle

df = pd.read_csv('BBFinalDataset.csv')

X = df.drop('GROUP',axis=1)
y = df['GROUP']

scaler = StandardScaler() 
scaler.fit(df.drop('GROUP',axis=1))

out = scaler.transform(df.drop('GROUP',axis=1))
df_scal = pd.DataFrame(out,columns=df.columns[:-1])

X_train, X_test, y_train, y_test = train_test_split(df_scal,df['GROUP'],test_size=0.30,random_state=101)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)

pickle.dump(knn, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

from werkzeug.wrappers import Request, Response
from flask import Flask, render_template, Response, request, redirect, url_for,jsonify     
from flask_cors import cross_origin
from firebase import firebase
import random

# app
app = Flask(__name__)

# routes
def home():
    return render_template("index.html")

def contestants(output):

    Dict = {1 : ["Oviya","Riythvika","Mugen","Losliya","Aari Arjunan","Tharshan","Raju","Ciby"], 2 : 
    ["Gayathri","Mahat","Vanitha","Archana","Niroop","Snehan","Aishwarya","Suresh","Abishek"], 3 :
    ["Ramya","Sandy","Gabriella","Velmurugan","Aajeedh","Isaivani","Chinnaponnu","Iykki","Imman"], 4 :
    ["Juliana","Sakthi","Madhumitha","Meera","Anitha","Vaishnavi","Ganja","Suja","Thamarai"], 5 :
    ["Aarava","Sanam","Ramya Pandian","Balaji Murugadoss","Rio","Priyanka","Varun","Shariq"], 6 :
    ["Anuya","Nadia","Anantha Vaithiyanadhan","Mamathi","Nithya","Mohan","Fathima","Suchitra","Vaiyapuri"], 7 :
    ["Ganesh","Balaji","Ponnambalam","Cheran","Saravanan","Ramesh","Abhinay Vaddi","Mathumitha Germany"], 8 :
    ["Namitha","Vijayalakshmi","Yashika","Mumtaz","Sherin","Sakshi","Kasthuri","Abhirami","Rekha"], 9 :
    ["Harish","Bindu","Raiza","Janani","Kavin","Shivani","Samyuktha","Akshara","Pavani"], 10 :
    ["Harathi","SomShekar","Suruthi","Bharani","Reshma","Sendrayan","Daniel","Kaajal","Nisha"]}

    r = Dict[output]
    return random.choice(r)


@app.route("/people/", methods=["GET","POST"])
def people():
    if request.method == "POST":
        Q1 = int(request.form['q1']);
        Q2 = int(request.form['q2']);
        Q3 = int(request.form['q3']);
        Q4 = int(request.form['q4']);
        Q5 = int(request.form['q5']);
        Q6 = int(request.form['q6']);        
        Q7 = int(request.form['q7']); 
        Q8 = int(request.form['q8']); 
        Q9 = int(request.form['q9']);
        Q10 = int(request.form['q10']);        
        Q11 = int(request.form['q11']); 
        
        arr = [Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11]; 
        result = contestants(knn.predict([arr])[0]);
        
    return render_template('index.html',text = result);


from werkzeug.serving import run_simple
#    firebase = firebase.FirebaseApplication("https://tamilnews-28a69-default-rtdb.firebaseio.com/",None)
run_simple('localhost', 9600, app)
