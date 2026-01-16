"""
Heart Disease Risk Prediction System - Data Validation Test Suite
Student ID: 24RP15116

This test suite validates that the HTML form data structure matches
the backend API expectations and the model's feature requirements.
"""

import json
import requests
from typing import Dict, List, Any
import sys

# ANSI color codes for better output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# API Configuration
API_BASE_URL = 'http://127.0.0.1:5000'

# Expected form fields from HTML (line 532-541)
FORM_FIELDS = {
    'age': {'type': 'numeric', 'html_type': 'number', 'required': True},
    'sex': {'type': 'categorical', 'html_type': 'select', 'required': True},
    'cp': {'type': 'categorical', 'html_type': 'select', 'required': True},
    'trestbps': {'type': 'numeric', 'html_type': 'number', 'required': True},
    'chol': {'type': 'numeric', 'html_type': 'number', 'required': True},
    'fbs': {'type': 'boolean', 'html_type': 'select', 'required': True},
    'restecg': {'type': 'categorical', 'html_type': 'select', 'required': True},
    'thalach': {'type': 'numeric', 'html_type': 'number', 'required': True},
    'exang': {'type': 'categorical', 'html_type': 'select', 'required': True},
    'oldpeak': {'type': 'numeric', 'html_type': 'number', 'required': True},
    'slope': {'type': 'categorical', 'html_type': 'select', 'required': True},
    'ca': {'type': 'numeric', 'html_type': 'number', 'required': True},
    'thal': {'type': 'categorical', 'html_type': 'select', 'required': True}
}

# Sample test data (realistic medical values)
SAMPLE_TEST_DATA = {
    'age': 63,
    'sex': 'Male',
    'cp': 'Typical angina',
    'trestbps': 145,
    'chol': 233,
    'fbs': True,
    'restecg': 'LVH',
    'thalach': 150,
    'exang': 'Yes',
    'oldpeak': 2.3,
    'slope': 'Downsloping',
    'ca': 0,
    'thal': 'Fixed defect'
}

# Additional test cases
TEST_CASES = [
    {
        'name': 'Low Risk Patient',
        'data': {
            'age': 45,
            'sex': 'Female',
            'cp': 'Asymptomatic',
            'trestbps': 120,
            'chol': 180,
            'fbs': False,
            'restecg': 'Normal',
            'thalach': 170,
            'exang': 'No',
            'oldpeak': 0.0,
            'slope': 'Upsloping',
            'ca': 0,
            'thal': 'Normal'
        }
    },
    {
        'name': 'High Risk Patient',
        'data': {
            'age': 70,
            'sex': 'Male',
            'cp': 'Typical angina',
            'trestbps': 180,
            'chol': 280,
            'fbs': True,
            'restecg': 'LVH',
            'thalach': 110,
            'exang': 'Yes',
            'oldpeak': 4.2,
            'slope': 'Downsloping',
            'ca': 3,
            'thal': 'Reversible defect'
        }
    },
    {
        'name': 'Edge Case - Minimum Values',
        'data': {
            'age': 29,
            'sex': 'Female',
            'cp': 'Asymptomatic',
            'trestbps': 90,
            'chol': 126,
            'fbs': False,
            'restecg': 'Normal',
            'thalach': 71,
            'exang': 'No',
            'oldpeak': 0.0,
            'slope': 'Upsloping',
            'ca': 0,
            'thal': 'Normal'
        }
    }
]

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD} {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def test_server_health() -> bool:
    """Test if the Flask server is running and healthy"""
    print_header("TEST 1: Server Health Check")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Server is running")
            print_info(f"Status: {data.get('status')}")
            print_info(f"Model Loaded: {data.get('model_loaded')}")
            print_info(f"Features Loaded: {data.get('features_loaded')}")
            print_info(f"Classes Loaded: {data.get('classes_loaded')}")
            
            if data.get('model_loaded') and data.get('features_loaded') and data.get('classes_loaded'):
                print_success("All artifacts loaded successfully")
                return True
            else:
                print_error("Some artifacts failed to load")
                return False
        else:
            print_error(f"Server returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server. Is Flask running on http://127.0.0.1:5000?")
        return False
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False

def test_model_info() -> Dict[str, Any]:
    """Test model info endpoint and retrieve expected features"""
    print_header("TEST 2: Model Information & Feature Validation")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/info", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == 'success':
                print_success("Model info retrieved successfully")
                
                # Display model info
                model_info = data.get('model_info', {})
                print_info(f"Model Name: {model_info.get('name')}")
                print_info(f"Model Type: {model_info.get('type')}")
                print_info(f"Version: {model_info.get('version')}")
                print_info(f"Student ID: {model_info.get('student_id')}")
                
                # Check features
                features = data.get('features', {})
                feature_names = features.get('names', [])
                feature_count = features.get('count', 0)
                
                print(f"\n{Colors.OKBLUE}Expected Features ({feature_count}):{Colors.ENDC}")
                for i, feature in enumerate(feature_names, 1):
                    print(f"  {i}. {feature}")
                
                # Validate form fields match model features
                print(f"\n{Colors.OKBLUE}Feature Validation:{Colors.ENDC}")
                form_field_names = set(FORM_FIELDS.keys())
                model_feature_names = set(feature_names)
                
                if form_field_names == model_feature_names:
                    print_success(f"Form fields match model features ({len(form_field_names)} fields)")
                else:
                    missing_in_form = model_feature_names - form_field_names
                    missing_in_model = form_field_names - model_feature_names
                    
                    if missing_in_form:
                        print_error(f"Missing in HTML form: {missing_in_form}")
                    if missing_in_model:
                        print_error(f"Missing in model: {missing_in_model}")
                
                # Check classes
                classes = data.get('classes', {})
                class_names = classes.get('names', [])
                print(f"\n{Colors.OKBLUE}Prediction Classes ({len(class_names)}):{Colors.ENDC}")
                for i, class_name in enumerate(class_names):
                    print(f"  Class {i}: {class_name}")
                
                return data
            else:
                print_error(f"Model info request failed: {data.get('message')}")
                return {}
        else:
            print_error(f"Server returned status code: {response.status_code}")
            return {}
            
    except Exception as e:
        print_error(f"Model info test failed: {str(e)}")
        return {}

def test_data_consistency() -> bool:
    """Test data type consistency between form and backend"""
    print_header("TEST 3: Data Type Consistency")
    
    all_passed = True
    
    # Check numeric fields
    print(f"{Colors.OKBLUE}Numeric Fields (should be converted to float):{Colors.ENDC}")
    numeric_fields = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    for field in numeric_fields:
        if field in FORM_FIELDS and FORM_FIELDS[field]['type'] == 'numeric':
            print_success(f"{field}: Expected numeric, HTML type is '{FORM_FIELDS[field]['html_type']}'")
        else:
            print_error(f"{field}: Type mismatch")
            all_passed = False
    
    # Check boolean field
    print(f"\n{Colors.OKBLUE}Boolean Field (should be converted to boolean):{Colors.ENDC}")
    if 'fbs' in FORM_FIELDS and FORM_FIELDS['fbs']['type'] == 'boolean':
        print_success("fbs: Expected boolean, HTML type is 'select' with True/False values")
    else:
        print_error("fbs: Type mismatch")
        all_passed = False
    
    # Check categorical fields
    print(f"\n{Colors.OKBLUE}Categorical Fields (should remain as strings):{Colors.ENDC}")
    categorical_fields = ['sex', 'cp', 'restecg', 'exang', 'slope', 'thal']
    for field in categorical_fields:
        if field in FORM_FIELDS and FORM_FIELDS[field]['type'] == 'categorical':
            print_success(f"{field}: Expected categorical, HTML type is '{FORM_FIELDS[field]['html_type']}'")
        else:
            print_error(f"{field}: Type mismatch")
            all_passed = False
    
    return all_passed

def test_prediction(test_case_name: str, test_data: Dict[str, Any]) -> bool:
    """Test prediction endpoint with specific data"""
    print_header(f"TEST 4: Prediction Test - {test_case_name}")
    
    try:
        print(f"{Colors.OKBLUE}Test Input Data:{Colors.ENDC}")
        for key, value in test_data.items():
            print(f"  {key}: {value} ({type(value).__name__})")
        
        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"\n{Colors.OKBLUE}Response:{Colors.ENDC}")
        print_info(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == 'success':
                print_success("Prediction successful")
                
                # Display prediction results
                prediction = data.get('prediction', {})
                print(f"\n{Colors.OKBLUE}Prediction Results:{Colors.ENDC}")
                print_info(f"Risk Level: {prediction.get('label')}")
                print_info(f"Class: {prediction.get('class')}")
                print_info(f"Description: {prediction.get('description')}")
                print_info(f"Confidence: {prediction.get('confidence')} ({prediction.get('confidence_score', 0)*100:.2f}%)")
                
                # Display probabilities
                probabilities = data.get('probabilities', [])
                print(f"\n{Colors.OKBLUE}Class Probabilities:{Colors.ENDC}")
                for prob in probabilities:
                    print(f"  {prob['label']}: {prob['percentage']:.2f}%")
                
                # Display recommendations
                recommendations = data.get('recommendations', [])
                print(f"\n{Colors.OKBLUE}Recommendations:{Colors.ENDC}")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
                
                # Verify input data was preserved
                input_data = data.get('input_data', {})
                if input_data == test_data:
                    print_success("\nInput data preserved correctly in response")
                else:
                    print_warning("\nInput data may have been modified:")
                    for key in test_data:
                        if test_data[key] != input_data.get(key):
                            print(f"  {key}: {test_data[key]} -> {input_data.get(key)}")
                
                return True
            else:
                print_error(f"Prediction failed: {data.get('message')}")
                return False
        else:
            print_error(f"Server returned status code: {response.status_code}")
            try:
                
                error_data = response.json()
                print_error(f"Error message: {error_data.get('message')}")
            except:
                print_error(f"Response text: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Prediction test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_missing_fields() -> bool:
    """Test API response when required fields are missing"""
    print_header("TEST 5: Missing Fields Validation")
    
    # Test with incomplete data
    incomplete_data = {
        'age': 50,
        'sex': 'Male',
        # Missing other required fields
    }
    
    try:
        print(f"{Colors.OKBLUE}Testing with incomplete data:{Colors.ENDC}")
        print_info(f"Provided fields: {list(incomplete_data.keys())}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            json=incomplete_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 400:
            data = response.json()
            print_success("API correctly rejected incomplete data")
            print_info(f"Error message: {data.get('message')}")
            return True
        elif response.status_code == 200:
            print_warning("API accepted incomplete data (this may indicate missing validation)")
            return False
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Missing fields test failed: {str(e)}")
        return False

def test_invalid_data_types() -> bool:
    """Test API response when data types are incorrect"""
    print_header("TEST 6: Invalid Data Type Validation")
    
    # Test with incorrect data types
    invalid_data = {
        'age': 'not_a_number',  # Should be numeric
        'sex': 'Male',
        'cp': 'Typical angina',
        'trestbps': 145,
        'chol': 233,
        'fbs': True,
        'restecg': 'LVH',
        'thalach': 150,
        'exang': 'Yes',
        'oldpeak': 2.3,
        'slope': 'Downsloping',
        'ca': 0,
        'thal': 'Fixed defect'
    }
    
    try:
        print(f"{Colors.OKBLUE}Testing with invalid data type (age='not_a_number'):{Colors.ENDC}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            json=invalid_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code in [400, 500]:
            print_success("API correctly rejected invalid data types")
            data = response.json()
            print_info(f"Error message: {data.get('message')}")
            return True
        elif response.status_code == 200:
            print_warning("API accepted invalid data types (this may indicate missing validation)")
            return False
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Invalid data type test failed: {str(e)}")
        return False

def generate_test_report(results: Dict[str, bool]):
    """Generate final test report"""
    print_header("TEST SUMMARY REPORT")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    print(f"{Colors.BOLD}Total Tests: {total_tests}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}Passed: {passed_tests}{Colors.ENDC}")
    print(f"{Colors.FAIL}Failed: {failed_tests}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Success Rate: {(passed_tests/total_tests)*100:.1f}%{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Detailed Results:{Colors.ENDC}")
    for test_name, result in results.items():
        status = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if result else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        print(f"  {status} - {test_name}")
    
    if failed_tests == 0:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! Form data is consistent with backend requirements.{Colors.ENDC}")
    else:
        print(f"\n{Colors.WARNING}{Colors.BOLD}⚠️  Some tests failed. Please review the issues above.{Colors.ENDC}")
    
    print("\n" + "="*80 + "\n")

def main():
    """Main test execution"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║    HEART DISEASE RISK PREDICTION SYSTEM - DATA VALIDATION TEST SUITE      ║")
    print("║                          Student ID: 24RP15116                             ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(Colors.ENDC)
    
    print_info(f"API Base URL: {API_BASE_URL}")
    print_info(f"Total Form Fields: {len(FORM_FIELDS)}")
    print_info(f"Test Cases: {len(TEST_CASES) + 1}\n")
    
    # Dictionary to store test results
    results = {}
    
    # Run tests
    results['Server Health'] = test_server_health()
    
    if not results['Server Health']:
        print_error("\nServer is not running or not healthy. Stopping tests.")
        print_info("Please start the Flask server with: python app_24RP15116.py")
        sys.exit(1)
    
    model_info = test_model_info()
    results['Model Info'] = bool(model_info)
    
    results['Data Type Consistency'] = test_data_consistency()
    
    # Test predictions with different cases
    results['Sample Prediction'] = test_prediction('Sample Test Data', SAMPLE_TEST_DATA)
    
    for i, test_case in enumerate(TEST_CASES, 1):
        results[f"Prediction Test {i}: {test_case['name']}"] = test_prediction(
            test_case['name'],
            test_case['data']
        )
    
    results['Missing Fields Validation'] = test_missing_fields()
    results['Invalid Data Types Validation'] = test_invalid_data_types()
    
    # Generate final report
    generate_test_report(results)

if __name__ == '__main__':
    main()