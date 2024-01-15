from django.http.response import JsonResponse
from rest_framework.decorators import api_view
# from injector import inject, Injector
# from ElectricityConsumptionForecast.dependencies.dependencies import AppModule
from ElectricityConsumptionForecastService.services.weather_data_service import WeatherService

# injector = Injector(modules=[AppModule()])

# @inject
@api_view(['GET'])
def get_head_of_data(request):
    weather_service = WeatherService()
    try:
        result = weather_service.get_head_of_data()
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)

# @inject
@api_view(['GET'])
def get_all(request):
    weather_service = WeatherService()
    try:
        result = weather_service.get_all()
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)

# @inject
@api_view(['POST'])
def save(request):
    weather_service = WeatherService()
    try:
        result = weather_service.save(request.FILES.getlist('rawForecastWeatherData'))
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400)

# @inject
@api_view(['DELETE'])
def delete_all(request):
    weather_service = WeatherService()
    try:
        result = weather_service.delete_all()
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)