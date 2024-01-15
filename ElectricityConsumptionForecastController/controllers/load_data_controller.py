from django.http.response import JsonResponse
from rest_framework.decorators import api_view
# from injector import inject, Injector
# from ElectricityConsumptionForecast.dependencies.dependencies import AppModule #configure
from ElectricityConsumptionForecastService.services.load_data_service import LoadService

# injector = Injector(modules=[AppModule()])
# configure()

# @inject
@api_view(['GET'])
def get_head_of_data(request):
    load_service = LoadService()
    try:
        result = load_service.get_head_of_data()
        return JsonResponse(result, status=200, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)

# @inject
@api_view(['GET'])
def get_all(request):
    load_service = LoadService()
    try:
        load_data = load_service.get_all()
        return JsonResponse(load_data, safe = False)
    except Exception as e:
        return JsonResponse({'Error Message': str(e)}, status=400, safe=False)

# @inject
@api_view(['POST'])
def save(request):
    load_service = LoadService()
    if load_service.save(request.FILES.getlist('files')):
        return JsonResponse({'Success Message': 'Data saved successfully'}, status=200)
    else:
        return JsonResponse({'Error'}, status=400, safe=False)