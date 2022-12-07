import pandas as pd


#Tuned Parameters
'''for i in ['Sphere','Rastrigin']:
    tune=pd.read_csv("/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/Tuning/"+i+"_Tuning.csv")
    summary=tune.agg({
        "w":["mean","median","min","max"],
        "c1":["mean","median","min","max"],
        "c2":["mean","median","min","max"]
    })
    summary.to_csv("/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/Summary/"+i+"_Tune_Summary.csv")
'''

#Perfromace Evalaution
for i in [30,60,90]:
    for j in ['Sphere','Rastrigin']:
        for k in ['Standard','Tuning']:
            data=pd.read_csv( "/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/N="+str(i)+"/"+str(j)+"_"+str(k)+"_Evalaution_Auto.csv")
            Iterations = data.agg({
                "iterations": ["mean", "median", "min", "max"],
                "success":["sum"]
            }).round(0)
            Eval=(Iterations.iloc[0,0]*i)/(Iterations.iloc[4,1]/200)
            Iterations["Evalaution"]=Eval

            Iterations.to_csv("/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/N="+str(i)+"/"+str(j)+"_"+str(k)+"Evalaution.csv")