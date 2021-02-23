from django.shortcuts import render
import joblib
# Create your views here.


def homeview(request):
    return render(request,'home.html')

def getPredictions(review):
    model = joblib.load('logClas_model.sav')
    tf = joblib.load('tfidf_model.sav')
    vect = joblib.load('vect.sav')

    ve = vect.transform([review])
    tfidf = tf.fit_transform(ve)
    prediction = model.predict(tfidf)

    if prediction == 0:
        return 0
    elif prediction == 1:
        return 1
    elif prediction == 2:
        return 2
    elif prediction == 3:
        return 3
    elif prediction == 4:
        return 4
    else:
        return 'error'

def resultview(request):
    review =  request.GET['review']
    res = getPredictions(review)
    return render(request,'result.html',{'result':res})