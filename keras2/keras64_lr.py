x = 0.5
y = 0.8      # 목표값
w = 0.5      # 가중치 초기값
lr = 0.01
epochs = 300

for i in range(epochs):
    predict = x * w
    loss = (predict - y) ** 2
    
    print("loss : ", round(loss, 4), "\tPredict : ", round(predict, 4))
    print("가중치 : ", round(w, 4), "\tepochs : ", epochs)
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2
    
    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr
    
    