
# Project Thesis
## Reconstruction of lost speech signal components using Machine Learning algorithms

This repository contains the code, data, and resources for the project thesis:  
**"Reconstruction of Lost Speech Signal Components Using Deep Learning Algorithms"**  
at Fachhochschule Dortmund University of Applied Sciences and Arts, Embedded Systems Engineering, 2025.


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-arch)
- [Installation](#installation)

## Overview

This project investigates speech signal enhancement and reconstruction using both classical and deep learning approaches. The main focus is on the MetricGAN architecture for speech enhancement, with comparison to the classical Wiener filter baseline.  
Objective metrics such as PESQ, STOI, and SNR are used for evaluation.

 
## Features

- Audio denoising and speech restoration with MetricGAN (pretrained, via [SpeechBrain](https://speechbrain.github.io/))
- Classical Wiener filtering implementation
- Evaluation using standard speech quality metrics
- Visualization of results (waveforms, spectrograms, and metric tables)
- User-friendly frontend (Svelte) and backend (FastAPI/Python).






## System Architecture

[Svelte Frontend] <------> [FastAPI Backend] <------> [SpeechBrain/MetricGAN, Wiener Filter, Metrics]

- Frontend: Svelte (interactive UI for audio upload and results)
- Backend: FastAPI (audio processing and inference)
- Speech Enhancement: MetricGAN (deep learning), Wiener filter (classical)
- Evaluation: PESQ, STOI, SNR



## Clone project

**Clone the repository:**
```bash
git clone https://github.com/yourusername/your-thesis-repo.git
cd project_thesis
```
   
## Installation using docker (Preferable)

**Navigate to the project directory.**
```bash
cd project/
```

**Run docker**
```bash
docker-compose up --build
```


The project will be avalable at http://localhost/

## Installation (Optional)

**Create a Python virtual environment.**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Install backend dependencies:**
```bash
pip install -r requirements.txt
```

**Install frontend dependencies:**
```bash
cd frontend
npm install
```

## Usage (When installed manually)
**Start the backend server:**
```bash
uvicorn app:app --reload
```


**Start the frontend (in frontend/):**
```bash
npm install
npm run build
```