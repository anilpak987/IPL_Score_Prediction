from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
app = Flask(__name__)

with open("dict.pickle", "rb") as f:
    batsman_bowler_dict,bat_bowl_team_dict,venue_dict = pickle.load(f) 

rf= pickle.load(open('rf.pkl', 'rb')) #for extracting model

#Extracting bastmen,bowlers and stadium names
batsmen_bowlers=list(batsman_bowler_dict.keys())
bat_bowl_team=list(bat_bowl_team_dict.keys())
match_venue=list(venue_dict.keys())

@app.route('/',methods=['GET'])
def Home():
    return render_template('ipl.html', Venues=match_venue,Bat_Bowl_Teams=bat_bowl_team,Batsmen_Bowlers=batsmen_bowlers)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        venue = request.form['venue']
        venue=venue_dict[venue]
        #print(venue)
        bat_team=request.form['bat_team']
        bat_team=bat_bowl_team_dict[bat_team]
        #print(bat_team)
        bowl_team = request.form['bowl_team']
        bowl_team=bat_bowl_team_dict[bowl_team]
        #print(bowl_team)
        batsman=request.form['batsman']
        batsman=batsman_bowler_dict[batsman]
        #print(batsman)
        bowler = request.form['bowler']
        bowler=batsman_bowler_dict[bowler]
        #print(bowler)
        runs = int(request.form['runs'])
        #print(runs)
        wickets=int(request.form['wickets'])
        #print(wickets)
        overs =float(request.form['overs'])
        #print(overs)
        runs_last_5=int(request.form['runs_last_5'])
        #print(runs_last_5)
        wickets_last_5 = int(request.form['wickets_last_5'])
        #print(wickets_last_5)
        output=rf.predict([[runs,wickets,overs,runs_last_5,wickets_last_5,venue,bat_team,bowl_team,batsman,bowler]])
        #print(output[0])
        return render_template('ipl.html',prediction_text="Predicted Score is : {}".format(output[0]))
    else:
        return render_template('ipl.html')
    
if __name__=="__main__":
    app.run()