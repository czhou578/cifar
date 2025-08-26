# CIFAR 100 Classification

This was a project of mine that was built using PyTorch to classify the images from the CIFAR-100 dataset. I attempted to build my own model without using any pretrained model from online, in an effort to improve my machine learning skills and understanding. The final model achieves 64% test accuracy. 

This is is a full stack web application being powered by FastAPI on the backend for serving the model and a React + TypeScript frontend. Drag and drop any image of your choice onto the frontend and see what the model predicted for you!

## Run Backend:

Navigate to the ```/backend``` folder and run

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Run Frontend:

Navigate to the ```/frontend``` folder and run

```
npm start
```


