library(forecast)
library(lmtest)
library(LSTS)
library(car)
library(neuralnet)

#input data
a=read.table(file.choose(),header=T)
rw=a$padi

#grafik persebaran
win.graph()
plot(rw,type="o",col="blue",
     ylab="Juta Ton",xlab="periode",lwd=1.5)
grid()
max(rw)
min(rw)

# Variabel Dummy
{jan=c(rep(c(1,rep(0,11)),5))
  feb=c(0,jan[-60])
  mar=c(0,feb[-60])
  apr=c(0,mar[-60])
  may=c(0,apr[-60])
  jun=c(0,may[-60])
  jul=c(0,jun[-60])
  aug=c(0,jul[-60])
  sep=c(0,aug[-60])
  oct=c(0,sep[-60])
  nov=c(0,oct[-60])
  dec=c(0,nov[-60])}
t=sequence(length(rw))
bulan=rep(sequence(12),5)
bulan=as.factor(bulan)
bulan=relevel(bulan,ref="4")
datadata=data.frame(padi=rw,t,jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec)
datadata

#tsr1
tsr1=lm(rw~t+bulan)
summary(tsr1)

#tsr2
tsr2=lm(rw~t+jan+feb+may+jun+jul+aug+sep+oct+nov+dec)
summary(tsr2)

#tsr3
tsr3=lm(rw~jan+feb+may+jun+jul+aug+sep+oct+nov+dec)
summary(tsr3)

#pemeriksaan diagnostik
ks.test(resid(tsr3),"pnorm",0,sd(resid(tsr3)))
Box.test(resid(tsr3),lag=12,type="Ljung")
Box.test(resid(tsr3),lag=24,type="Ljung")
Box.test(resid(tsr3),lag=36,type="Ljung")
Box.test(resid(tsr3),lag=48,type="Ljung")
Box.test(resid(tsr3),lag=59,type="Ljung")
Box.Ljung.Test(resid(tsr3),59)

#plot acf pacf untuk penambahan lag tsr
win.graph()
acf(resid(tsr3),59)
grid()
pacf(resid(tsr3),59)
grid()

#TSR3+lag
forlag=data.frame(rw,jan,feb,mar,apr,may,jun
                  ,jul,aug,sep,oct,nov,dec)
forlag
lag2=rw[23:58]
lag24=rw[1:36]
tsrlag=lm(rw~jan+feb+may+jun+jul+aug+sep+oct+nov+dec+lag2+lag24,data=forlag[25:60,])
summary(tsrlag)

#pemeriksaan diagnostik tsr lag
ks.test(resid(tsrlag),"pnorm",0,sd(resid(tsrlag)))
Box.test(resid(tsrlag),lag=6,type="Ljung")
Box.test(resid(tsrlag),lag=12,type="Ljung")
Box.test(resid(tsrlag),lag=18,type="Ljung")
Box.test(resid(tsrlag),lag=24,type="Ljung")
Box.test(resid(tsrlag),lag=30,type="Ljung")
Box.test(resid(tsrlag),lag=35,type="Ljung")
Box.Ljung.Test(resid(tsrlag),35)

#prediksi TSR
ptsr=fitted(tsrlag)
ptsr
rw[25:60]
#peramalan TSR
tsrlag$coefficients
l2=rw[59:60]
l2
l24=rw[37:48]
l24
ztopi=c(NA)
ztopi
for(i in 1:12){
  fc=(13.763-9.678*jan[i]-7.709*feb[i]-3.635*may[i]-4.972*jun[i]
      -4.853*jul[i]-4.935*aug[i]-4.951*sep[i]-6.280*oct[i]
      -7.667*nov[i]-9.693*dec[i]-0.317*l2[i]-0.515*l24[i])
  l2=c(l2,fc)
  ztopi=c(ztopi,fc)
}
ztopi=ztopi[-1]
# HASIL PERAMALAN TSR
round(ztopi,3)
# PREDIKSI DAN PERAMALAN TSR
prediksi_peramalan_tsr=data.frame(c(ptsr,ztopi))
round(prediksi_peramalan_tsr,3)

#MAPE TSR TERBAIK
MAPETSR=mean(abs(resid(tsrlag)*100/rw[25:60]))
MAPETSR

#grafik prediksi peramalan
win.graph()
awal=c(rw[25:60],rep(NA,12))
periode=seq(from=25, to=72,by=1)
plot(periode,awal,
     xlab="Periode",ylab="Produksi Padi (Juta Ton)",
     xlim=c(25,72),ylim=c(0,11))
lines(periode,awal)
points(periode[1:36],ptsr,col="red")
lines(periode[1:36],ptsr,col="red")
points(periode[37:48],ztopi,col="green")
lines(periode[36:48],c(ptsr[36],ztopi),col="green")
grid()
legend(55, 11.3, legend=c("Aktual", "Prediksi","Peramalan"),
       col=c("black", "red","green"), lty=1, cex=1)


##Pembentukan nn
acf(resid(tsrlag),35)
grid()

pacf(resid(tsrlag),35)
grid()

error=resid(tsrlag)
error

# DATA MASUKAN NN
epadi=data.frame(Y=c(error[13:36]),X1=c(error[1:24]))
round(epadi,3)
match(max(error),error)
match(min(error),error)


# HASIL STANDARDISASI DATA MASUKAN NN
max=max(error)
min=min(error)
max
min
epadi1=((epadi-min)*1.8/(max-min))+(-0.9)
epadi1

# Pelatihan NN 1 Neuron
{set.seed(1)
  nn1=neuralnet(Y~X1,data=epadi1,hidden=1,act.fct="tanh",threshold=0.05,algorithm="backprop",
                stepmax=10000,startweights=NULL,learningrate=0.01,linear.output=T)}
# Pelatihan NN 2 Neuron
{set.seed(1)
  nn2=neuralnet(Y~X1,data=epadi1,hidden=2,act.fct="tanh",threshold=0.05,algorithm="backprop",
                stepmax=10000,startweights=NULL,learningrate=0.01,linear.output=T)}
# Pelatihan NN 3 Neuron
{set.seed(1)
  nn3=neuralnet(Y~X1,data=epadi1,hidden=3,act.fct="tanh",threshold=0.05,algorithm="backprop",
                stepmax=10000,startweights=NULL,learningrate=0.01,linear.output=T)}

# Prediksi NN 1 neuron tersembunyi
nn1
plot(nn1)
Predictnn1=as.data.frame(nn1$net.result)
predn1=Predictnn1$s
predn1

# HASIL DESTANDARDISASI PREDIKSI NN 1 NEURON TERSEMBUNYI
prednn1=((predn1+0.9)*(max-min)/1.8)+min
prednn1

# Prediksi NN 2 neuron tersembunyi
nn2
plot(nn2)
Predictnn2=as.data.frame(nn2$net.result)
predn2=Predictnn2$s
# HASIL DESTANDARDISASI PREDIKSI NN 2 NEURON TERSEMBUNYI
prednn2=((predn2+0.9)*(max-min)/1.8)+min
prednn2

# prediksi NN 3 neuron tersembunyi
nn3
plot(nn3)
Predictnn3=as.data.frame(nn3$net.result)
predn3=Predictnn3$s
# HASIL DESTANDARDISASI PREDIKSI NN 3 NEURON TERSEMBUNYI
prednn3=((predn3+0.9)*(max-min)/1.8)+min
prednn3

# Hasil Prediksi NN 1, 2, dan 3 Neuron
hasil_prediksi_nn=data.frame(NN_1_Neuron=prednn1,
                             NN_2_Neuron=prednn2,
                             NN_3_Neuron=prednn3)
hasil_prediksi_nn

##PEMODELAN HYBRID
#TSR-NN1
predtsr=fitted(tsrlag)
predtsr
tsrnn1=predtsr[13:36]+prednn1
tsrnn1
NN1=c(rep(NA,12),prednn1)
NN1
TSRNN1=c(rep(NA,12),tsrnn1)
TSRNN1
tabelbantu=data.frame(tsr=predtsr,NN1,TSRNN1) 
round(tabelbantu,3)

#TSR-NN2
predtsr=fitted(tsrlag)
predtsr
tsrnn2=predtsr[13:36]+prednn2
tsrnn2
NN2=c(rep(NA,12),prednn2)
NN2
TSRNN2=c(rep(NA,12),tsrnn2)
TSRNN2
tabelbantu2=data.frame(tsr=predtsr,NN2,TSRNN2) 
round(tabelbantu2,3)

#TSR-NN3
predtsr=fitted(tsrlag)
predtsr
tsrnn3=predtsr[13:36]+prednn3
tsrnn3
NN3=c(rep(NA,12),prednn3)
NN3
TSRNN3=c(rep(NA,12),tsrnn3)
TSRNN3
tabelbantu3=data.frame(tsr=predtsr,NN3,TSRNN3) 
round(tabelbantu3,6)


#MAPE TSR-NN1
MAPETSRNN1=mean(abs((tsrnn1-rw[37:60])*100/rw[37:60]))
MAPETSRNN1

#MAPE TSR-NN2
MAPETSRNN2=mean(abs((tsrnn2-rw[37:60])*100/rw[37:60]))
MAPETSRNN2

#MAPE TSR-NN3
MAPETSRNN3=mean(abs((tsrnn3-rw[37:60])*100/rw[37:60]))
MAPETSRNN3

## peramalan NN1
Ytest=c(epadi1$Y,rep(0,12))
Ytest
for (i in 25:36){ 
  Xtest=t(matrix(c(Ytest[i-12]),byrow=FALSE))
  Ytest[i]=compute(nn1,covariate=Xtest)$net.result
}
Ytest=as.data.frame(Ytest[25:36])
Ytest
Ytest=((Ytest+0.9)*(max-min)/1.8)+min
round(Ytest,3)
peramalan=(ztopi+Ytest)
round(peramalan,3)

#grafik prediksi peramalan
win.graph()
awal=c(rw[25:60],rep(NA,12))
periode=seq(from=25, to=72,by=1)
plot(periode,awal,
     xlab="Periode",ylab="Produksi Padi (Juta Ton)",
     xlim=c(25,72),ylim=c(0,11))
lines(periode,awal)
points(periode[1:36],c(rep(NA,12),tsrnn1),col="red")
lines(periode[1:36],c(rep(NA,12),tsrnn1),col="red")
points(periode[37:48],peramalan$`Ytest[25:36]`,col="green")
lines(periode[36:48],c(tsrnn1[24],peramalan$`Ytest[25:36]`),col="green")
grid()
legend(55, 11.3, legend=c("Aktual", "Prediksi","Peramalan"),
       col=c("black", "red","green"), lty=1, cex=1)

#grafik perbandingan TSR dan Hybrid TSR-NN
win.graph()
awal=c(rw[25:60],rep(NA,12))
periode=seq(from=25, to=72,by=1)
plot(periode,awal,
     xlab="Periode",ylab="Produksi Padi (Juta Ton)",
     xlim=c(25,72),ylim=c(0,11))
lines(periode,awal)
points(periode[1:48],c(rep(NA,12),tsrnn1,peramalan$`Ytest[25:36]`),col="blue")
lines(periode[1:48],c(rep(NA,12),tsrnn1,peramalan$`Ytest[25:36]`),col="blue")
points(periode[1:48],c(ptsr,ztopi),col="red")
lines(periode[1:48],c(ptsr,ztopi),col="red")

grid()
legend(53, 11.3, legend=c("Aktual", "Model TSR","Model Hybrid TSR-NN 1 Neuron"),
       col=c("black", "red","blue"), lty=1, cex=0.8)

