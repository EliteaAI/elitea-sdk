import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
from typing import Any

from elitea_sdk.runtime.clients.prompt import EliteAPrompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class TestEliteAPrompt:
    """Test suite for EliteAPrompt class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_elitea = Mock()
        self.mock_elitea.predict.return_value = [
            AIMessage(content="response1"),
            AIMessage(content="response2")
        ]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "Hello {name}, please help with {task}")
        ])
        
        self.llm_settings = {"temperature": 0.7, "max_tokens": 100}
        
        self.elitea_prompt = EliteAPrompt(
            elitea=self.mock_elitea,
            prompt=self.prompt,
            name="test_prompt",
            description="Test prompt description",
            llm_settings=self.llm_settings
        )
    
    def test_init(self):
        """Test EliteAPrompt initialization"""
        assert self.elitea_prompt.elitea == self.mock_elitea
        assert self.elitea_prompt.prompt == self.prompt
        assert self.elitea_prompt.name == "test_prompt"
        assert self.elitea_prompt.description == "Test prompt description"
        assert self.elitea_prompt.llm_settings == self.llm_settings
    
    def test_create_pydantic_model_with_variables(self):
        """Test creating Pydantic model with prompt variables"""
        Model = self.elitea_prompt.create_pydantic_model()
        
        # Should include all prompt variables plus 'input'
        expected_fields = {'name', 'task', 'input'}
        assert set(Model.__fields__.keys()) == expected_fields
        
        # All fields should be string type with None default
        for field_name, field_info in Model.__fields__.items():
            assert field_info.annotation == str
            assert field_info.default is None
    
    def test_create_pydantic_model_no_variables(self):
        """Test creating Pydantic model with no prompt variables"""
        simple_prompt = ChatPromptTemplate.from_messages([("human", "Hello")])
        simple_elitea_prompt = EliteAPrompt(
            elitea=self.mock_elitea,
            prompt=simple_prompt,
            name="simple",
            description="Simple prompt",
            llm_settings={}
        )
        
        Model = simple_elitea_prompt.create_pydantic_model()
        
        # Should only have 'input' field
        assert set(Model.__fields__.keys()) == {'input'}
    
    def test_create_pydantic_model_with_input_variable(self):
        """Test creating Pydantic model when 'input' is already a variable"""
        prompt_with_input = ChatPromptTemplate.from_messages([
            ("human", "Process this {input} with {parameter}")
        ])
        elitea_prompt_with_input = EliteAPrompt(
            elitea=self.mock_elitea,
            prompt=prompt_with_input,
            name="test",
            description="Test",
            llm_settings={}
        )
        
        Model = elitea_prompt_with_input.create_pydantic_model()
        
        # Should have both 'input' and 'parameter' fields, no duplicates
        expected_fields = {'input', 'parameter'}
        assert set(Model.__fields__.keys()) == expected_fields
    
    def test_predict_with_variables(self):
        """Test predict method with variables"""
        variables = {
            "name": "John",
            "task": "write code",
            "input": "user input here"
        }
        
        result = self.elitea_prompt.predict(variables)
        
        # Check that elitea.predict was called correctly
        self.mock_elitea.predict.assert_called_once()
        call_args = self.mock_elitea.predict.call_args
        
        messages, llm_settings, elitea_vars = call_args[0][0], call_args[0][1], call_args[1]['variables']
        
        # Check messages structure
        assert len(messages) == 3  # system + human template + user input
        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == "user input here"
        
        # Check LLM settings
        assert llm_settings == self.llm_settings
        
        # Check variables format
        expected_vars = [
            {"name": "name", "value": "John"},
            {"name": "task", "value": "write code"}
        ]
        assert elitea_vars == expected_vars
        
        # Check result format
        assert result == "response1\n\nresponse2"
    
    def test_predict_without_variables(self):
        """Test predict method without variables"""
        result = self.elitea_prompt.predict()
        
        # Check that elitea.predict was called with empty variables
        call_args = self.mock_elitea.predict.call_args
        elitea_vars = call_args[1]['variables']
        assert elitea_vars == []
        
        # Check that user input is empty
        messages = call_args[0][0]
        assert messages[-1].content == ''
    
    def test_predict_with_none_variables(self):
        """Test predict method with None variables"""
        result = self.elitea_prompt.predict(None)
        
        # Should behave same as no variables
        call_args = self.mock_elitea.predict.call_args
        elitea_vars = call_args[1]['variables']
        assert elitea_vars == []
    
    def test_predict_with_empty_input(self):
        """Test predict method with empty input"""
        variables = {"name": "Alice", "task": "help"}
        
        result = self.elitea_prompt.predict(variables)
        
        call_args = self.mock_elitea.predict.call_args
        messages = call_args[0][0]
        
        # Input should be empty string when not provided
        assert messages[-1].content == ''
    
    def test_predict_with_only_input(self):
        """Test predict method with only input variable"""
        variables = {"input": "just user input"}
        
        result = self.elitea_prompt.predict(variables)
        
        call_args = self.mock_elitea.predict.call_args
        messages, elitea_vars = call_args[0][0], call_args[1]['variables']
        
        # No elitea variables should be passed
        assert elitea_vars == []
        
        # Input should be passed as user message
        assert messages[-1].content == "just user input"
    
    def test_predict_single_response(self):
        """Test predict with single response"""
        self.mock_elitea.predict.return_value = [AIMessage(content="single response")]
        
        result = self.elitea_prompt.predict({})
        
        assert result == "single response"
    
    def test_predict_empty_response(self):
        """Test predict with empty response"""
        self.mock_elitea.predict.return_value = []
        
        result = self.elitea_prompt.predict({})
        
        assert result == ""
    
    def test_predict_multiple_responses(self):
        """Test predict with multiple responses"""
        self.mock_elitea.predict.return_value = [
            AIMessage(content="response1"),
            AIMessage(content="response2"),
            AIMessage(content="response3")
        ]
        
        result = self.elitea_prompt.predict({})
        
        assert result == "response1\n\nresponse2\n\nresponse3"
    
    def test_predict_with_special_characters(self):
        """Test predict with special characters in variables"""
        variables = {
            "name": "José María",
            "task": "handle UTF-8 çharacters & symbols!",
            "input": "unicode test: 🚀 emoji"
        }
        
        result = self.elitea_prompt.predict(variables)
        
        # Should handle special characters without errors
        call_args = self.mock_elitea.predict.call_args
        elitea_vars = call_args[1]['variables']
        
        assert any(var['value'] == "José María" for var in elitea_vars)
        assert any(var['value'] == "handle UTF-8 çharacters & symbols!" for var in elitea_vars)
    
    def test_predict_with_numeric_values(self):
        """Test predict with numeric values in variables"""
        variables = {
            "name": "123",  # Convert to string
            "task": "45.67",  # Convert to string  
            "input": "True"  # Convert to string
        }
        
        # Should not raise errors
        result = self.elitea_prompt.predict(variables)
        
        call_args = self.mock_elitea.predict.call_args
        elitea_vars = call_args[1]['variables']
        
        # Values should be strings
        assert any(var['value'] == "123" for var in elitea_vars)
        assert any(var['value'] == "45.67" for var in elitea_vars)
    
    def test_predict_message_combination(self):
        """Test that messages are properly combined"""
        variables = {
            "name": "Test",
            "task": "verify",
            "input": "user message"
        }
        
        self.elitea_prompt.predict(variables)
        
        call_args = self.mock_elitea.predict.call_args
        messages = call_args[0][0]
        
        # Should have original prompt messages + user input
        assert len(messages) == len(self.prompt.messages) + 1
        
        # Last message should be user input
        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == "user message"
        
        # Previous messages should be from the prompt
        for i, original_msg in enumerate(self.prompt.messages):
            assert type(messages[i]) == type(original_msg)
    
    def test_predict_elitea_error_handling(self):
        """Test error handling when elitea.predict raises exception"""
        self.mock_elitea.predict.side_effect = Exception("EliteA service error")
        
        with pytest.raises(Exception, match="EliteA service error"):
            self.elitea_prompt.predict({})
    
    def test_predict_response_with_empty_content(self):
        """Test predict with response containing empty content"""
        self.mock_elitea.predict.return_value = [
            AIMessage(content=""),
            AIMessage(content="real content"),
            AIMessage(content="")
        ]
        
        result = self.elitea_prompt.predict({})
        
        assert result == "\n\nreal content\n\n"
