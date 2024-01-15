from django.http.response import JsonResponse
from rest_framework.decorators import api_view
# from injector import inject, Injector
# from ElectricityConsumptionForecast.dependencies.dependencies import AppModule
from ElectricityConsumptionForecastService.services.train_data_service import TrainService

# injector = Injector(modules=[AppModule()])

# @inject
@api_view(['POST'])
def train_with_regression(request):
    train_service = TrainService()
    try:
        result = train_service.train_with_regression(request.data.get('startDate'), request.data.get('endDate'))
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)

# @inject
@api_view(['POST'])
def train_with_neural_network(request):
    train_service = TrainService()
    try:
        result = train_service.train_with_neural_network(request.data.get('startDate'), request.data.get('endDate'))
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)
