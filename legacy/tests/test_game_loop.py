import pytest
import pygame
from unittest.mock import MagicMock, patch
from game.game_loop import GameLoop, GameState
import logging

# Disable logging for tests
logging.getLogger("smartcrossing.game").setLevel(logging.CRITICAL)

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

def test_initialization():
    """Test that GameLoop properly initializes pygame and display"""
    game = GameLoop()
    
    # Check that pygame and display are initialized
    assert pygame.get_init()
    assert pygame.display.get_init()
    
    # Check that screen was created
    assert game.screen is not None
    assert isinstance(game.screen, pygame.Surface)
    
    # Check that game state was initialized
    assert game.state is not None
    assert isinstance(game.state, GameState)
    
    # Check that game objects were created
    assert len(game.dynamic_objects) > 0
    assert len(game.static_objects) > 0

@patch('pygame.event.get')
def test_handle_events_reinitializes_display(mock_event_get):
    """Test that handle_events reinitializes display if needed"""
    # Set up
    mock_event_get.return_value = []  # No events
    game = GameLoop()
    
    # Force display to be uninitialized
    if pygame.display.get_init():
        pygame.display.quit()
    
    # Handle events should reinitialize display
    game.handle_events()
    
    # Check that display was reinitialized
    assert pygame.display.get_init()
    assert game.screen is not None

@patch('pygame.event.get')
def test_quit_event_handling(mock_event_get):
    """Test that quit events are handled properly"""
    # Set up mock quit event
    mock_event = MagicMock()
    mock_event.type = pygame.QUIT
    mock_event_get.return_value = [mock_event]
    
    game = GameLoop()
    game.handle_events()
    
    # Check that game state is set to not running
    assert not game.state.running
    
    # Check that pygame is still initialized (cleanup happens in run())
    assert pygame.get_init()

def test_cleanup_on_exception():
    """Test that resources are cleaned up even if an exception occurs"""
    game = GameLoop()
    
    # Force an exception in the game loop
    with pytest.raises(Exception):
        with patch.object(game, 'update', side_effect=Exception("Test exception")):
            game.run()
    
    # Check that pygame and display were properly cleaned up
    assert not pygame.get_init()
    assert not pygame.display.get_init()

def test_render_reinitializes_display():
    """Test that render reinitializes display if needed"""
    game = GameLoop()
    
    # Force display to be uninitialized
    if pygame.display.get_init():
        pygame.display.quit()
    
    # Render should reinitialize display
    game.render()
    
    # Check that display was reinitialized
    assert pygame.display.get_init()
    assert game.screen is not None 