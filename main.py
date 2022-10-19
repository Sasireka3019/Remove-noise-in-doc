from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from sklearn.model_selection import train_test_split
# build the autoencoder
def get_model():
  In = Input(shape=(32,32,1))
  x = Conv2D(64,(3,3),activation="relu", padding="same")(In)
  x = MaxPooling2D((2,2), padding="same")(x)
  x = Conv2D(32,(3,3),activation="relu", padding="same")(x)
  x = MaxPooling2D((2,2), padding="same")(x)
  x = Conv2D(32,(3,3),activation="relu", padding="same")(x)
  x = UpSampling2D((2,2))(x)
  x = Conv2D(64,(3,3),activation="relu", padding="same")(x)
  x = UpSampling2D((2,2))(x)
  Out = Conv2D(1,(3,3),activation="sigmoid", padding="same")(x)
  model = Model(In,Out)
  model.compile(optimizer="adam",loss="binary_crossentropy")
  return model
def clean_util():
  model = get_model()
  X = []
  Y = []
  for files in zip(train_files,train_cleaned_files):
    img = cv2.imread(files[0],cv2.IMREAD_GRAYSCALE)/255.
    X.append(img)
    img = cv2.imread(files[1],cv2.IMREAD_GRAYSCALE)/255.
    Y.append(img)
  X = np.array(X)
  Y = np.array(Y)
  X = X.reshape(-1,32,32,1)
  Y = Y.reshape(-1,32,32,1)
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
  model.fit(x_train,y_train, epochs=10, batch_size=16, validation_data=(x_test,y_test))
def get_chunks(file):
  page = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
  oph, opw=page.shape[:2]  
  nph, npw = oph-(oph%32),opw-(opw%32) 
  row_chunks=nph//32 
  col_chunks=npw//32 
  rc=0 # row counter 
  cc=0 # column counter 
  img_chunks = np.ones((row_chunks,col_chunks,32,32,1),dtype="float32")
  for row in range(0,nph,32):
    cc=0
    for col in range(0,npw,32):
      nimg = page[row:row+32,col:col+32]/255.
      nimg =np.array(nimg).reshape(32,32,1)
      try:
        img_chunks[rc,cc]=nimg
      except:
        print(rc,cc)
      cc+=1
    rc+=1
  return img_chunks
def show_chunks(chunks):
  for row in chunks:
    plt.figure(figsize=(10,10))
    for i,chunk in enumerate(row):
      plt.subplot(1,len(row),i+1)
      plt.imshow(chunk.reshape(32,32),"gray")
      plt.axis("OFF")
    plt.show()
def reassemble_chunks(chunks):
  oph, opw=chunks.shape[0]*32,chunks.shape[1]*32    
  the_page = np.ones((oph,opw),dtype="float32")
  for r, row in enumerate(chunks):
    r=r*32
    for c, chunk in enumerate(row):
      c=c*32
      the_page[r:r+32,c:c+32]=chunk.reshape(32,32)       
  return the_page
 def clean_page(doc_file):
  doc = fitz.open(doc_file)
  for file in doc:
    test_img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    test_chunks = get_chunks(file)
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.imshow(test_img,"gray")
    pred_chunks = model.predict(test_chunks.reshape(-1,32,32,1))
    pred_chunks = pred_chunks.reshape(test_chunks.shape)
    the_page = reassemble_chunks(pred_chunks)
    plt.subplot(1,2,2)
    plt.imshow(the_page,"gray")
    plt.show()
if __name__ == "__main__":
  clean_util()
  clean_page("demo.pdf")
