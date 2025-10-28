"""
data_processor.py - Course Data Processing Module

This module is responsible for taking your raw JSON course data and transforming it
into a structured format that the rest of the system can use efficiently.

Think of this as the "data preparation kitchen" - it takes raw ingredients (JSON)
and prepares them into a form that's ready to be "cooked" (processed by AI).
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, time
import re
import logging

# Set up logging so you can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CourseDataProcessor:
    """
    Main class for processing GT course data.
    
    This class handles:
    1. Loading raw JSON data from your partner's pipeline
    2. Normalizing and cleaning the data
    3. Creating computed fields for better search
    4. Handling prerequisites parsing
    5. Time slot standardization
    """
    
    def __init__(self, json_file_path: str):
        """
        Initialize the processor with a path to your course JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing course data
                           (e.g., "data/raw/gt_courses.json")
        """
        self.courses_df = None  # Will hold our processed DataFrame
        self.json_file_path = json_file_path
        
        # Load and process the data immediately
        self.load_and_process_data()
        
        logger.info(f"Successfully loaded {len(self.courses_df)} courses")
    
    def load_and_process_data(self):
        """
        Main processing pipeline - this is where the magic happens!
        
        The JSON should have a structure like:
        {
            "courses": [
                {
                    "course_number": "CS 1301",
                    "course_name": "Introduction to Computing",
                    "description": "...",
                    "professor": "Dr. Smith",
                    "location": "Klaus 1443",
                    "times": {
                        "days": "MW",
                        "start_time": "09:30",
                        "end_time": "10:45"
                    },
                    "prerequisites": ["MATH 1551"],
                    "credits": 3,
                    "semester": "Fall 2024"
                },
                ...
            ]
        }
        """
        logger.info(f"Loading course data from {self.json_file_path}")
        
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {self.json_file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file: {e}")
            raise
        
        # Extract courses (handle different JSON structures)
        if isinstance(raw_data, dict) and 'courses' in raw_data:
            courses_list = raw_data['courses']
        elif isinstance(raw_data, list):
            courses_list = raw_data
        else:
            raise ValueError("JSON must be either a list of courses or dict with 'courses' key")
        
        # Convert to DataFrame for easier manipulation
        self.courses_df = pd.DataFrame(courses_list)
        
        # Process each aspect of the data
        self._normalize_course_numbers()
        self._parse_time_slots()
        self._process_prerequisites()
        self._add_computed_columns()
        self._validate_data()
        
        logger.info("Data processing complete!")
    
    def _normalize_course_numbers(self):
        """
        Standardize course numbers to a consistent format.
        
        Examples:
        - "CS1301" -> "CS 1301"
        - "cs 1301" -> "CS 1301"
        - "CS-1301" -> "CS 1301"
        """
        logger.info("Normalizing course numbers...")
        
        def normalize_course_number(course_num: str) -> str:
            if pd.isna(course_num):
                return ""
            
            # Remove any non-alphanumeric characters except spaces
            course_num = re.sub(r'[^\w\s]', ' ', str(course_num))
            
            # Extract department code and number
            match = re.match(r'([A-Za-z]+)\s*(\d+)', course_num)
            if match:
                dept = match.group(1).upper()
                num = match.group(2)
                return f"{dept} {num}"
            
            return course_num.upper()
        
        self.courses_df['course_number'] = self.courses_df['course_number'].apply(
            normalize_course_number
        )
    
    def _parse_time_slots(self):
        """
        Parse and standardize time information.
        
        Converts various time formats into consistent datetime objects.
        Handles formats like:
        - "9:30 AM" -> time(9, 30)
        - "14:00" -> time(14, 0)
        - "2:00 PM" -> time(14, 0)
        """
        logger.info("Parsing time slots...")
        
        def parse_time(time_str: str) -> Optional[time]:
            """Convert time string to datetime.time object"""
            if pd.isna(time_str) or not time_str:
                return None
            
            time_str = str(time_str).strip()
            
            # Try different time formats
            formats = [
                '%H:%M',      # 24-hour format: "14:30"
                '%I:%M %p',   # 12-hour with AM/PM: "2:30 PM"
                '%I:%M%p',    # 12-hour without space: "2:30PM"
                '%H:%M:%S',   # With seconds: "14:30:00"
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(time_str, fmt)
                    return dt.time()
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse time: {time_str}")
            return None
        
        # Handle nested time information
        if 'times' in self.courses_df.columns:
            # If times is a nested dict/object
            times_df = pd.json_normalize(self.courses_df['times'])
            self.courses_df = pd.concat([self.courses_df, times_df], axis=1)
            self.courses_df.drop('times', axis=1, inplace=True)
        
        # Parse start and end times
        if 'start_time' in self.courses_df.columns:
            self.courses_df['start_time'] = self.courses_df['start_time'].apply(parse_time)
        if 'end_time' in self.courses_df.columns:
            self.courses_df['end_time'] = self.courses_df['end_time'].apply(parse_time)
        
        # Standardize days of the week
        if 'days' in self.courses_df.columns:
            # Convert to uppercase and remove spaces
            self.courses_df['days'] = self.courses_df['days'].fillna('').str.upper().str.replace(' ', '')
    
    def _process_prerequisites(self):
        """
        Process prerequisite information into a consistent format.
        
        Handles various formats:
        - String: "CS 1301, CS 1331" -> ["CS 1301", "CS 1331"]
        - List: ["CS 1301", "CS 1331"] -> ["CS 1301", "CS 1331"]
        - Complex: "CS 1301 and (CS 1331 or CS 1332)" -> ["CS 1301", "CS 1331 or CS 1332"]
        """
        logger.info("Processing prerequisites...")
        
        def parse_prerequisites(prereq_data) -> List[str]:
            """Parse prerequisites into a list of course numbers"""
            if pd.isna(prereq_data) or not prereq_data:
                return []
            
            if isinstance(prereq_data, list):
                return [str(p).strip() for p in prereq_data]
            
            if isinstance(prereq_data, str):
                # Split by common delimiters
                prereqs = re.split(r'[,;]', prereq_data)
                
                # Clean up each prerequisite
                cleaned = []
                for prereq in prereqs:
                    prereq = prereq.strip()
                    if prereq:
                        # Try to extract course numbers
                        courses = re.findall(r'[A-Z]+\s*\d+', prereq.upper())
                        if courses:
                            cleaned.extend(courses)
                        elif prereq:  # Keep as-is if no pattern match
                            cleaned.append(prereq)
                
                return cleaned
            
            return []
        
        self.courses_df['prerequisites'] = self.courses_df.get('prerequisites', []).apply(
            parse_prerequisites
        )
    
    def _add_computed_columns(self):
        """
        Add helpful computed columns that make searching and filtering easier.
        
        These columns are derived from existing data but make our life easier later.
        """
        logger.info("Adding computed columns...")
        
        # Extract course level (1000, 2000, 3000, 4000)
        def get_course_level(course_num: str) -> int:
            """Extract the thousand-level of the course"""
            match = re.search(r'(\d)', str(course_num))
            if match:
                return int(match.group(1)) * 1000
            return 0
        
        self.courses_df['level'] = self.courses_df['course_number'].apply(get_course_level)
        
        # Create a searchable text field combining all relevant information
        # This makes our embedding search more effective
        def create_searchable_text(row) -> str:
            """Combine all searchable fields into one text block"""
            parts = []
            
            # Add course number and name
            parts.append(f"{row.get('course_number', '')} {row.get('course_name', '')}")
            
            # Add description
            if pd.notna(row.get('description')):
                parts.append(str(row['description']))
            
            # Add professor name (helps when students search for specific professors)
            if pd.notna(row.get('professor')):
                parts.append(f"Professor: {row['professor']}")
            
            # Add topics/keywords if available
            if pd.notna(row.get('topics')) and row.get('topics'):
                if isinstance(row['topics'], list):
                    parts.append(' '.join(row['topics']))
                else:
                    parts.append(str(row['topics']))
            
            return ' '.join(parts)
        
        self.courses_df['searchable_text'] = self.courses_df.apply(
            create_searchable_text, axis=1
        )
        
        # Add time period classification (morning/afternoon/evening)
        def classify_time_period(start_time) -> str:
            """Classify course into time period for easier filtering"""
            if pd.isna(start_time) or not start_time:
                return 'unknown'
            
            hour = start_time.hour
            if hour < 12:
                return 'morning'
            elif hour < 17:
                return 'afternoon'
            else:
                return 'evening'
        
        self.courses_df['time_period'] = self.courses_df.get('start_time', pd.Series()).apply(
            classify_time_period
        )
        
        # Calculate course duration in minutes
        def calculate_duration(row) -> int:
            """Calculate class duration in minutes"""
            if pd.notna(row.get('start_time')) and pd.notna(row.get('end_time')):
                start = row['start_time']
                end = row['end_time']
                if start and end:
                    # Convert to minutes since midnight for calculation
                    start_minutes = start.hour * 60 + start.minute
                    end_minutes = end.hour * 60 + end.minute
                    return end_minutes - start_minutes
            return 0
        
        self.courses_df['duration_minutes'] = self.courses_df.apply(
            calculate_duration, axis=1
        )
    
    def _validate_data(self):
        """
        Validate the processed data and report any issues.
        
        This helps catch problems early before they cause issues downstream.
        """
        logger.info("Validating processed data...")
        
        # Check for required columns
        required_columns = ['course_number', 'course_name', 'description']
        missing_columns = set(required_columns) - set(self.courses_df.columns)
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
        
        # Check for missing values in critical columns
        for col in ['course_number', 'course_name']:
            if col in self.courses_df.columns:
                missing_count = self.courses_df[col].isna().sum()
                if missing_count > 0:
                    logger.warning(f"{missing_count} missing values in {col}")
        
        # Check for duplicate course numbers
        if 'course_number' in self.courses_df.columns:
            duplicates = self.courses_df['course_number'].duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Found {duplicates} duplicate course numbers")
        
        # Report summary statistics
        logger.info(f"Data validation complete:")
        logger.info(f"  - Total courses: {len(self.courses_df)}")
        logger.info(f"  - Columns: {list(self.courses_df.columns)}")
        logger.info(f"  - Course levels: {self.courses_df['level'].value_counts().to_dict()}")
        if 'time_period' in self.courses_df.columns:
            logger.info(f"  - Time periods: {self.courses_df['time_period'].value_counts().to_dict()}")
    
    def save_processed_data(self, output_path: str):
        """
        Save the processed DataFrame to disk for later use.
        
        Args:
            output_path: Where to save the processed data (e.g., "data/processed/courses.pkl")
        """
        logger.info(f"Saving processed data to {output_path}")
        self.courses_df.to_pickle(output_path)
        logger.info(f"Saved {len(self.courses_df)} courses to {output_path}")
    
    def get_sample_courses(self, n: int = 5) -> pd.DataFrame:
        """
        Get a sample of courses for testing/debugging.
        
        Args:
            n: Number of courses to sample
            
        Returns:
            DataFrame with n random courses
        """
        return self.courses_df.sample(n=min(n, len(self.courses_df)))
    
    def get_courses_by_level(self, level: int) -> pd.DataFrame:
        """
        Get all courses at a specific level (e.g., 3000-level courses).
        
        Args:
            level: Course level (1000, 2000, 3000, or 4000)
            
        Returns:
            DataFrame with courses at that level
        """
        return self.courses_df[self.courses_df['level'] == level]
    
    def search_courses(self, query: str) -> pd.DataFrame:
        """
        Simple text search across course data (not semantic - just keyword matching).
        
        Args:
            query: Search query
            
        Returns:
            DataFrame with matching courses
        """
        query = query.lower()
        mask = self.courses_df['searchable_text'].str.lower().str.contains(query, na=False)
        return self.courses_df[mask]


# Example usage and testing
if __name__ == "__main__":
    # This code runs when you execute the file directly
    # It's useful for testing the data processor independently
    
    # Initialize processor with your data
    processor = CourseDataProcessor('data/raw/gt_courses.json')
    
    # Show sample courses
    print("\n=== Sample Courses ===")
    sample = processor.get_sample_courses(3)
    for _, course in sample.iterrows():
        print(f"{course['course_number']}: {course['course_name']}")
        print(f"  Professor: {course.get('professor', 'TBA')}")
        print(f"  Time: {course.get('days', 'TBA')} {course.get('start_time', 'TBA')}")
        print(f"  Prerequisites: {course.get('prerequisites', [])}")
        print()
    
    # Test search functionality
    print("\n=== Search Test ===")
    results = processor.search_courses("machine learning")
    print(f"Found {len(results)} courses matching 'machine learning'")
    
    # Save processed data
    processor.save_processed_data('data/processed/courses.pkl')
    print("\nData saved successfully!")
