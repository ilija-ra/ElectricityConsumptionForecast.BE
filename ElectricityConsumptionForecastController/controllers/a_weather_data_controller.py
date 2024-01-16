from django.http.response import JsonResponse
from rest_framework.decorators import api_view
# from injector import inject, Injector
# from ElectricityConsumptionForecast.dependencies.dependencies import AppModule
from ElectricityConsumptionForecastService.services.a_weather_data_service import AWeatherService

# injector = Injector(modules=[AppModule()])

# @inject
@api_view(['GET'])
def a_get_head_of_data(request):
    weather_service = AWeatherService()
    try:
        result = weather_service.a_get_head_of_data()
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)

# @inject
@api_view(['GET'])
def a_get_all(request):
    weather_service = AWeatherService()
    try:
        result = weather_service.a_get_all()
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)

# @inject
@api_view(['POST'])
def a_save(request):
    weather_service = AWeatherService()
    try:
        result = weather_service.a_save(request.FILES.getlist('rawForecastWeatherData'))
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400)

# @inject
@api_view(['DELETE'])
def a_delete_all(request):
    weather_service = AWeatherService()
    try:
        result = weather_service.a_delete_all()
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)