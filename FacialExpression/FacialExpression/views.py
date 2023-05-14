from django.http import HttpResponse
from django.shortcuts import render, redirect
from Project import TestData
from django.urls import reverse
import os
import random

def home(request):
    found = request.GET.get('found')
    if found == "No Face detected":
        return render(request, "home.html", {"Found": False})
    return render(request, "home.html")

def emotion(request):

    if(request.GET.get('mood')):
       emo = request.GET.get('mood')
    else:
        emo = TestData.Test()
    if emo == "No Face detected":
        return redirect('/?found=' + emo)

    path= "\\Music\\" + emo
    img_url = "\\Images\\" + emo + ".jpg"
    files=os.listdir(".\\static"+path)
    d=random.choice(files)
    song_url = path + "\\" + d
    
    return render(request, "Emotion.html",{"emo": emo, "song_url": song_url, "song_name": d, "img_url": img_url})
