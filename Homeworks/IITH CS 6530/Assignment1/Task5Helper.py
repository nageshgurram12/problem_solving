# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:47:38 2018

@author: nrgurram
"""

in_data = open("./dataset/user-shows.txt", "r")
out_data = open("./dataset/sample_out.txt", "w")
userIndex=0;
for line in in_data:
    movies = line.split(" ");
    userIndex+=1;
    movieIndex = 0;
    for movie in movies:
        movieIndex+=1;
        movie = movie.strip();
        if movie == "1":
            out_data.write(str(userIndex) + "," + str(movieIndex) + "\n");

in_data.close();
out_data.close();