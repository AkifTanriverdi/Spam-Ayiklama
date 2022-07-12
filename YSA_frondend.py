from tkinter import*
from tkinter import messagebox
from django.contrib.auth import authenticate, login

def login():
    mail=entry1.get()

    if(mail==""):
        messagebox.showinfo("","Bu Alan Boş Olamaz!")
    else:
        import time
        import pickle
        import tensorflow as tf
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # only use GPU memory that we need, not allocate all the GPU memory / tüm GPU belleğini tahsis etmeyin, yalnızca ihtiyacımız olan GPU belleğini kullanın
            tf.config.experimental.set_memory_growth(gpus[0], enable=True)

        import tqdm
        import numpy as np
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.metrics import Recall, Precision 
        import keras.metrics as keras_metrics
        
        
        #parametrelerin tanımlanması
        SEQUENCE_LENGTH = 100 # the length of all sequences (number of words per sample) / tüm dizilerin uzunluğu (örnek başına kelime sayısı)
        EMBEDDING_SIZE = 100  # Using 100-Dimensional GloVe embedding vectors / 100 Boyutlu GloVe gömme vektörlerini kullanma
        TEST_SIZE = 0.25 # ratio of testing set / test seti oranı
        BATCH_SIZE = 64
        EPOCHS =1 #mber of epochs
        label2int = {"ham": 0, "spam": 1}
        int2label = {0: "ham", 1: "spam"}


        """
        Loads SMS Spam Collection dataset
        SMS İstenmeyen Posta Toplama veri kümesini yükler
        """
        def load_data(): 
            texts, labels = [], []
            with open("C:/Users/akift/Desktop/YSA_spam/data/smsspamcollection/SMSSpamCollection") as f:
                for line in f:
                    split = line.split()
                    labels.append(split[0].strip())
                    texts.append(' '.join(split[1:]).strip())
            return texts, labels
        # load the data / verileri yükle
        X, y = load_data()


        #Tokenizer ile veri setinin hazırlanması
        # Text tokenization / Metin belirteci
        # vectorizing text, turning each text into sequence of integers / metni vektörleştirme, her metni tamsayı dizisine dönüştürme
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)
        # lets dump it to a file, so we can use it in testing /bir dosyaya atalım, böylece testte kullanabiliriz
        pickle.dump(tokenizer, open("C:/Users/akift/Desktop/YSA_spam/data/results/tokenizer.pickle", "wb"))
        # convert to sequence of integers / tamsayı dizisine dönüştürme.
        X = tokenizer.texts_to_sequences(X)
        print(X[0])
        #sabit uzunlukta bir diziye sahip olmak için  pad_sequences() fonksiyonunun kullanılması.
        # convert to numpy arrays / numpy dizilerine dönüştür
        X = np.array(X)
        y = np.array(y)
        # pad sequences at the beginning of each sequence with 0's / 0'lı her dizinin başındaki pad dizileri
        X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

        print(X[0])

        # One Hot encoding labels /kodlama etiketleri
        y = [ label2int[label] for label in y ]
        y = to_categorical(y)

        print(y[0])

        #eğitim ve test verilerini karıştırıp bölelim.
        # split and shuffle / böl ve karıştır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=7)
        # print our data shapes / veri şekillerimizi yazdır
        print("X_train.shape:", X_train.shape)
        print("X_test.shape:", X_test.shape)
        print("y_train.shape:", y_train.shape)
        print("y_test.shape:", y_test.shape)


               #Modeli oluşturmaya başlıyoruz.
               #Önceden eğitilmiş gömme vektörlerini yüklemek için bir fonksiyon yazıyoruz.
        def get_embedding_vectors(tokenizer, dim=100):
            embedding_index = {}
            with open(f"C:/Users/akift/Desktop/YSA_spam/data/glove.6B.{dim}d.txt", encoding='utf8') as f:
                for line in tqdm.tqdm(f, "Reading GloVe"):
                    values = line.split()
                    word = values[0]
                    vectors = np.asarray(values[1:], dtype='float32')
                    embedding_index[word] = vectors

            word_index = tokenizer.word_index
            embedding_matrix = np.zeros((len(word_index)+1, dim))
            for word, i in word_index.items():
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    # words not found will be 0s / bulunamayan kelimeler 0 olacak
                    embedding_matrix[i] = embedding_vector
                    
            return embedding_matrix


         #Modeli oluşturan işlevi(LSTM) tanımlayalım
        def get_model(tokenizer, lstm_units):
           
            '''Constructs the model,
            Embedding vectors => LSTM => 2 output Fully-Connected neurons with softmax activation
            
            Modeli oluşturur,
            Vektörleri gömme => LSTM => 2 çıktı softmax aktivasyonlu Tam Bağlantılı nöronlar
            '''
            # get the GloVe embedding vectors / GloVe gömme vektörlerini alın
            embedding_matrix = get_embedding_vectors(tokenizer)
            model = Sequential()
            model.add(Embedding(len(tokenizer.word_index)+1,
                      EMBEDDING_SIZE,
                      weights=[embedding_matrix],
                      trainable=False,
                      input_length=SEQUENCE_LENGTH))

            model.add(LSTM(lstm_units, recurrent_dropout=0.2))
            model.add(Dropout(0.3))
            model.add(Dense(2, activation="softmax"))
            # compile as rmsprop optimizer / rmsprop iyileştirici olarak derleyin
            # aswell as with recall metric /  geri çağırma metriği ile

            model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                          metrics=["accuracy", keras_metrics.Precision(), keras_metrics.Recall()])
            model.summary()
            return model

        # constructs the model with 128 LSTM units / 128 LSTM birimli modeli oluşturur
        model = get_model(tokenizer=tokenizer, lstm_units=128)

         #modeli yeni yüklediğimiz verilerle eğitmemiz gerekiyor.
        # initialize our ModelCheckpoint and TensorBoard callbacks / ModelCheckpoint ve TensorBoard geri aramalarımızı başlatın
        # model checkpoint for saving best weights / en iyi ağırlıkları kaydetmek için model kontrol noktası
        model_checkpoint = ModelCheckpoint("results/spam_classifier_{val_loss:.2f}.h5", save_best_only=True,
                                            verbose=1)
        # for better visualization / daha iyi görselleştirme için
        tensorboard = TensorBoard(f"logs/spam_classifier_{time.time()}")
        
        # train the model / modeli eğit
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  batch_size=BATCH_SIZE, epochs=EPOCHS,
                  callbacks=[tensorboard, model_checkpoint],
                  verbose=1)

         #Modelimizin değerlendirilmesi.
        # get the loss and metrics / kaybı ve metrikleri al
        result = model.evaluate(X_test, y_test)
        # extract those / bunları çıkar
        loss = result[0]
        accuracy = result[1]
        precision = result[2]
        recall = result[3]

        print(f"[+] Accuracy: {accuracy*100:.2f}%")
        print(f"[+] Precision:   {precision*100:.2f}%")
        print(f"[+] Recall:   {recall*100:.2f}%")

       #MOdelin test edilmesi
        def get_predictions(text):
            sequence = tokenizer.texts_to_sequences([text])
            # pad the sequence / sırayı doldur
            sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
            # get the prediction / tahmini al
            prediction = model.predict(sequence)[0]
            # one-hot encoded vector, revert using np.argmax / tek sıcak kodlanmış vektör, np.argmax kullanarak geri alma1
            return int2label[np.argmax(prediction)]


        '''text1 = "I saw your profile on the internet and wanted to reach you! You may be well-suited to many of the remote software engineering roles that top US companies hire at Turing.'''
        '''text2=You can activate your membership from the "CREATE MEMBERSHIP" button below to use your e-mail address with 50 TL defined.
100% FRE€SPN FOR SWEET BONANZA
100.000TL AWARD SPORTS TOURNAMENT
DRAWING IN 15 MINUTES!'''



        ##print(get_predictions(mail))
        messagebox.showinfo(get_predictions(mail))
        
root=Tk()
root.title("login")
root.geometry("500x400")

global entry1


Label(root,text="Email Giriniz").place(x=50,y=20)
 
entry1=Entry(root,bd=5)
entry1.place(x=130,y=20)

Button(root,text="Kontrol Et",command=login,height=1,width=13,bd=10).place(x=135,y=100)



root.mainloop()