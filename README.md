# LPG Calibration
This repo is a set of tools that I use for calibrating LPG autogas system (KME Diego G3) in my car.

It also contains example data used for processing by these tools.


Currently this repo is a work in progress - a project that I develop in my free time (that I don't have much of).

It's not really well documented now, as for now I treat it mostly just as a way of versioning my notes and progress on this subject.

I will do my best to document it and make it more usable for others - however I don't really expect anyone to benefit from it much, as the LPG system (KME Diego G3) is now rather old, no longer mounted in cars and was never really popular outside of middle Europe (at least I do believe so).

If you're interested in this project, feel free to DM me.

## Overview
For now this repo contains an OCR tool that creates a .csv file from screen recording of a Diego G3 application. 

The file contains parameters like Gas Injection Time, Gasoline injection time, RPM, temperature sensors, air pressure and Gas pressure.

It can later be used in another tool that merges data logged by the "CarScanner" OBD reader(mainly STFT-Short term fuel trim and LTFT-Long term fuel trim) with data logged from Gas ECU, and then can draw a graph of dependency between RPM, Intake Manifold Pressure and Fuel Trims to fine tune LPG system in those ranges.

Currently I'm working on a better solution - a data logger application for either raspberryPi or maybe some kind of Arduino/ESP32 that would directly connect to the gas ECU via serial connection and create .csv log from the serial communication.

It would help with not having the need to drive around with a laptop in a car with OBS started and recording.

As of now, I'm still decoding the communication protocol.