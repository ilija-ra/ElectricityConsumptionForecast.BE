# from injector import inject, Binder
# from ElectricityConsumptionForecastService.services import weather_data_service, load_data_service, preprocess_data_service
# from ElectricityConsumptionForecastRepository.repositories import weather_data_repository, load_data_repository, preprocess_data_repository

# @inject
# def configure(binder: Binder):
#     binder.bind(weather_data_service.WeatherService, to=weather_data_service.WeatherService)
#     binder.bind(load_data_service.LoadService, to=load_data_service.LoadService)
#     binder.bind(weather_data_repository.WeatherRepository, to=weather_data_repository.WeatherRepository)
#     binder.bind(load_data_repository.LoadRepository, to=load_data_repository.LoadRepository)


from injector import inject, Module
from ElectricityConsumptionForecastService.services import weather_data_service, load_data_service
from ElectricityConsumptionForecastRepository.repositories import weather_data_repository, load_data_repository

class AppModule(Module):
    @inject
    def configure(self, binder):
        binder.bind(weather_data_service.WeatherService, to=weather_data_service.WeatherService)
        binder.bind(load_data_service.LoadService, to=load_data_service.LoadService)
        binder.bind(weather_data_repository.WeatherRepository, to=weather_data_repository.WeatherRepository)
        binder.bind(load_data_repository.LoadRepository, to=load_data_repository.LoadRepository)