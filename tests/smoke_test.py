import unittest
from app import app

class TestAppSmoke(unittest.TestCase):
	def setUp(self):
		app.testing = True
		self.client = app.test_client()
	
	# Complete the function below to test a success in running the application
	# Tests that app starts and the page responds with status 200
	def test_prediction_route_success(self):
		response = self.client.get('/')
		self.assertEqual(response.status_code, 200, f'Page responded with status {response.status_code} when status 200 expected, check that the app starts and the ‘/’ route in app.py > home()')
		

	# Complete the function below to test a form is rendered
	# Tests that form is rendered with all its features
	def test_get_form(self):
		response = self.client.get('/')

		# Tests that form is rendered
		self.assertIn(b'<form', response.data, 'Form is missing from response, check templates/form.html')

		# Tests that temperature input field is rendered
		self.assertIn(b'name="temperature"', response.data, 'Temperature input field is missing from form, check templates/form.html')

		# Tests that pressure input field is rendered
		self.assertIn(b'name="pressure"', response.data, 'Pressure input field is missing from form, check templates/form.html')

		# Tests that humidity input field is rendered
		self.assertIn(b'name="humidity"', response.data, 'Humidity input field is missing from form, check templates/form.html')

		# Tests that wind speed input field is rendered
		self.assertIn(b'name="wind_speed"', response.data, 'Wind speed input field is missing from form, check templates/form.html')

		# Tests that wind degree input field is rendered
		self.assertIn(b'name="wind_deg"', response.data, 'Wind degree input field is missing from form, check templates/form.html')

		# Tests that rain 1h input field is rendered
		self.assertIn(b'name="rain_1h"', response.data, 'Rain 1h input field is missing from form, check templates/form.html')

		# Tests that rain 3h input field is rendered
		self.assertIn(b'name="rain_3h"', response.data, 'Rain 3h input field is missing from form, check templates/form.html')

		# Tests that snow input field is rendered
		self.assertIn(b'name="snow"', response.data, 'Snow input field is missing from form, check templates/form.html')

		# Tests that clouds input field is rendered
		self.assertIn(b'name="clouds"', response.data, 'Clouds input field is missing from form, check templates/form.html')

		# Tests that submit button is rendered
		self.assertIn(b'type="submit"', response.data, 'Submit button is missing from form, check templates/form.html')
		
 
if __name__ == '__main__':
	unittest.main()
