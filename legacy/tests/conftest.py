import pytest
import pygame
import logging
import os

# Disable logging for tests
logging.getLogger("smartcrossing.game").setLevel(logging.CRITICAL)
logging.getLogger("smartcrossing.rl").setLevel(logging.CRITICAL)

@pytest.fixture(autouse=True)
def cleanup_pygame():
    """Fixture to clean up pygame before and after each test"""
    # Setup
    if pygame.get_init():
        pygame.quit()
    
    yield  # This is where the test runs
    
    # Teardown
    if pygame.get_init():
        pygame.quit()

@pytest.fixture
def test_models_dir():
    """Fixture to create and clean up a test models directory"""
    os.makedirs('test_models', exist_ok=True)
    yield 'test_models'
    # Cleanup
    if os.path.exists('test_models'):
        for file in os.listdir('test_models'):
            os.remove(os.path.join('test_models', file))
        os.rmdir('test_models') 