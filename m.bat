@echo off
FOR /L %%A IN (1,1,200) DO (
  python livecamera.py
  timeout 3
)