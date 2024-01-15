from django.apps import AppConfig

class ElectricityconsumptionforecastcontrollerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ElectricityConsumptionForecastController'
    # def ready(self):
    #     from injector import Injector
    #     from ElectricityConsumptionForecast.dependencies.dependencies import AppModule
    #     from ElectricityConsumptionForecastService.services.weather_data_service import WeatherService
    #     # Create an instance of AppModule to configure bindings
    #     app_module = AppModule()

    #     # Create an Injector instance with the configured bindings
    #     injector = Injector(modules=[app_module])

    #     # You can store the injector instance for later use if needed
    #     self.injector = injector
