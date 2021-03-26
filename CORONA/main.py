from flask import Flask, render_template,request

app = Flask(__name__)
import pickle
import flask
file=open('model.pkl','rb')

clf=pickle.load(file)
@app.route('/',methods=["GET","POST"])
def entry_point():
     if request.method =="POST":
          myDict=request.form
          fever=int(myDict['fever'])
          Age=int(myDict['Age'])
          pain=int(myDict['pain'])
          runnyNose=int(myDict['runnyNose'])
          diffBreath=int(myDict['diffBreath'])
          InputFeatures=[fever,pain,Age,runnyNose,diffBreath]
          infProb=clf.predict_proba([InputFeatures])[0][1] 
          print(infProb)
          return render_template('show.html',inf=round(infProb*100))
     return render_template('index.html')

     #return 'Hello World!' + str(infProb)

if __name__ == '__main__':
    app.run(debug=True)