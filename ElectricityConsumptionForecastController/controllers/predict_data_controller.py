from django.http.response import JsonResponse
from rest_framework.decorators import api_view
# from injector import inject, Injector
# from ElectricityConsumptionForecast.dependencies.dependencies import AppModule
from ElectricityConsumptionForecastService.services.predict_data_service import PredictService

# injector = Injector(modules=[AppModule()])

# @inject
@api_view(['POST'])
def predict_with_regression(request):
    predict_service = PredictService()
    try:
        result = predict_service.predict_with_regression(request.data.get('startDate'), request.data.get('numberOfDays'))
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)

# @inject
@api_view(['POST'])
def predict_with_neural_network(request):
    predict_service = PredictService()
    try:
        result = predict_service.predict_with_neural_network(request.data.get('startDate'), request.data.get('numberOfDays'))
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)
    
@api_view(['GET'])
def get_prediction_result(request):
    predict_service = PredictService()
    try:
        result = predict_service.get_prediction_result()
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)