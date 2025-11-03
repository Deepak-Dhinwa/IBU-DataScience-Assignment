"""
Visualization Agent - Generates Plotly charts from quantitative data.
Extracts data and creates 1-3 important visualizations.
"""
import json
import re
import plotly.graph_objects as go
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..models import llm


def run_viz_agent(analysis_data: str) -> tuple[str, list]:
    """
    Visualization Agent that generates 1-3 important Plotly charts from quantitative data.
    Returns: (description_text, list_of_plotly_figures)
    """
    try:
        print(f"Running Visualization Agent...")
        if llm is None:
            return "Error: Base LLM is not loaded.", []
        
        # Step 1: Extract quantitative data in structured format
        extracted_json = _extract_quantitative_data(analysis_data)
        
        # Step 2: Parse JSON
        data = _parse_extracted_data(extracted_json)
        if not data:
            return """**Visualization Not Possible**

No quantitative data found in the analysis. Visualization requires numbers, percentages, or measurable metrics.""", []
        
        if not data.get("has_data", False):
            return """**Visualization Not Possible**

No quantitative data found in the analysis. Visualization requires numbers, percentages, or measurable metrics.""", []
        
        # Step 3: Generate 1-3 important visualizations
        charts, description_parts = _generate_charts(data)
        
        if not charts:
            return """**Visualization Not Possible**

Extracted data exists but cannot be visualized with current chart types.
Data may need different visualization approach.""", []
        
        description = "**Generated Visualizations**\n\n" + "\n".join(description_parts)
        description += f"\n\n*Generated {len(charts)} chart(s) from quantitative data in the analysis.*"
        
        return description, charts
    except Exception as e:
        print(f"Error in Visualization Agent: {e}")
        import traceback
        traceback.print_exc()
        return f"Error in Visualization Agent: {e}", []


def _extract_quantitative_data(analysis_data: str) -> str:
    """Extract quantitative data using LLM."""
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract quantitative data in structured JSON format for visualization.

**Extract:**
- Numbers, percentages, counts
- Categories with values (e.g., "Homework: 10%")
- Time series data (if dates present)
- Comparisons (if mentioned)

**Output (JSON format):**
```json
{{
  "has_data": true/false,
  "data_points": [
    {{"label": "Category1", "value": 10, "type": "percentage/number/count"}},
    {{"label": "Category2", "value": 30, "type": "percentage"}}
  ],
  "time_series": [
    {{"date": "2024-01-01", "value": 5}},
    {{"date": "2024-02-01", "value": 8}}
  ] (optional, only if dates exist),
  "comparisons": [
    {{"item1": "X", "item2": "Y", "difference": 15}}
  ] (optional)
}}
```

**Rules:**
- If NO quantitative data, set "has_data": false
- Extract ONLY explicit numbers
- Preserve exact values from analysis

**Analysis:**
{analysis}"""),
        ("user", "Extract quantitative data as JSON. If no data exists, return {{\"has_data\": false}}.")
    ])
    
    extract_chain = extract_prompt | llm | StrOutputParser()
    return extract_chain.invoke({"analysis": analysis_data})


def _parse_extracted_data(extracted_json: str):
    """Parse JSON from LLM response."""
    try:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', extracted_json, re.DOTALL)
        if not json_match:
            # Try without json tag
            json_match = re.search(r'```\s*(\{.*?\})\s*```', extracted_json, re.DOTALL)
        if not json_match:
            # Try to find any JSON object
            json_match = re.search(r'\{.*\}', extracted_json, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
            # Clean up the JSON string - remove trailing commas, fix common issues
            json_str = json_str.strip()
            # Remove trailing commas before closing braces/brackets
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            parsed_data = json.loads(json_str)
            print(f"Successfully parsed JSON data: {list(parsed_data.keys())}")
            return parsed_data
        else:
            # Try to parse the entire string
            parsed_data = json.loads(extracted_json)
            return parsed_data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Attempted to parse: {extracted_json[:200]}...")
        # Fallback: check if it says no data
        if "NO_QUANTITATIVE_DATA" in extracted_json or "has_data\": false" in extracted_json.lower() or "\"has_data\": false" in extracted_json.lower():
            return None
        return None
    except Exception as e:
        print(f"Error in visualization data extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_charts(data: dict) -> tuple[list, list]:
    """Generate Plotly charts from extracted data."""
    charts = []
    description_parts = []
    
    # Chart 1: If we have category data points
    if data.get("data_points") and len(data["data_points"]) > 0:
        try:
            labels = []
            values = []
            for item in data["data_points"]:
                label = item.get("label", item.get("name", "Unknown"))
                value = item.get("value", item.get("amount", item.get("count", 0)))
                # Try to convert to float
                try:
                    value = float(value) if isinstance(value, (int, float, str)) else 0
                    if value > 0:  # Only add valid positive values
                        labels.append(str(label))
                        values.append(value)
                except (ValueError, TypeError):
                    continue
            
            if labels and values and len(labels) == len(values):
                # Choose chart type based on data
                if len(labels) <= 8 and all(v >= 0 for v in values):
                    # Pie chart for proportions
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                    fig.update_layout(title="Distribution Analysis", showlegend=True)
                    charts.append(fig)
                    description_parts.append("**Chart 1: Distribution** - Shows proportion of categories")
                else:
                    # Bar chart for comparisons
                    fig = go.Figure(data=[go.Bar(x=labels, y=values)])
                    fig.update_layout(title="Category Comparison", xaxis_title="Category", yaxis_title="Value")
                    charts.append(fig)
                    description_parts.append("**Chart 1: Category Comparison** - Shows values across categories")
        except Exception as e:
            print(f"Error generating data_points chart: {e}")
    
    # Chart 2: Time series if available
    if data.get("time_series") and len(data["time_series"]) > 1:
        try:
            dates = []
            values = []
            for item in data["time_series"]:
                date = item.get("date", item.get("time", item.get("month", "")))
                value = item.get("value", item.get("amount", item.get("total", 0)))
                # Try to convert to float
                try:
                    value = float(value) if isinstance(value, (int, float, str)) else 0
                    if date and value > 0:  # Only add valid entries
                        dates.append(str(date))
                        values.append(value)
                except (ValueError, TypeError):
                    continue
            
            if dates and values and len(dates) == len(values) and len(dates) > 1:
                fig = go.Figure(data=[go.Scatter(x=dates, y=values, mode='lines+markers')])
                fig.update_layout(title="Trend Over Time", xaxis_title="Date", yaxis_title="Value")
                charts.append(fig)
                description_parts.append("**Chart 2: Trend Over Time** - Shows temporal progression")
        except Exception as e:
            print(f"Error generating time_series chart: {e}")
    
    # Chart 3: Comparisons if available
    if data.get("comparisons") and len(data["comparisons"]) > 0:
        try:
            items = []
            differences = []
            for comp in data["comparisons"][:5]:  # Max 5 comparisons
                item1 = comp.get('item1', comp.get('label1', ''))
                item2 = comp.get('item2', comp.get('label2', ''))
                diff = comp.get("difference", comp.get("diff", comp.get("value", 0)))
                try:
                    diff = float(diff) if isinstance(diff, (int, float, str)) else 0
                    if item1 or item2:
                        items.append(f"{item1} vs {item2}")
                        differences.append(diff)
                except (ValueError, TypeError):
                    continue
            
            if items and differences and len(items) == len(differences):
                fig = go.Figure(data=[go.Bar(x=items, y=differences)])
                fig.update_layout(title="Comparison Analysis", xaxis_title="Comparison", yaxis_title="Difference")
                charts.append(fig)
                description_parts.append("**Chart 3: Comparison** - Shows differences between entities")
        except Exception as e:
            print(f"Error generating comparisons chart: {e}")
    
    # Limit to 3 most important charts
    return charts[:3], description_parts[:3]

