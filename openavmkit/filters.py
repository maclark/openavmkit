import pandas as pd


def select_filter(df: pd.DataFrame, f: list) -> pd.DataFrame:
  """
  Select a subset of the DataFrame based on a list of filters.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param f: Filter expressed as a list.
  :type f: list
  :returns: Filtered DataFrame.
  :rtype: pandas.DataFrame
  """
  resolved_index = resolve_filter(df, f)
  return df.loc[resolved_index]


def resolve_bool_filter(df: pd.DataFrame, f: list) -> pd.Series:
  """
  Resolve a list of filters using a boolean operator.

  Iterates through each filter in the list (after the operator) and combines their boolean indices
  using the specified boolean operator ("and", "or", "nand", "nor", "xor", "xnor").

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param f: List where the first element is the boolean operator and the remaining elements are filter objects.
  :type f: list
  :returns: Boolean Series resulting from applying the boolean operator.
  :rtype: pandas.Series
  """
  operator = f[0]
  values = f[1:]

  final_index = None

  for v in values:
    selected_index = resolve_filter(df, v)

    if final_index is None:
      final_index = selected_index
      continue

    if operator == "and":
      final_index = final_index & selected_index
    elif operator == "nand":
      final_index = ~(final_index & selected_index)
    elif operator == "or":
      final_index = final_index | selected_index
    elif operator == "nor":
      final_index = ~(final_index | selected_index)
    elif operator == "xor":
      final_index = final_index ^ selected_index
    elif operator == "xnor":
      final_index = ~(final_index ^ selected_index)

  return final_index


def resolve_filter(df: pd.DataFrame, f: list) -> pd.Series:
  """
  Resolve a filter list into a boolean Series for the DataFrame (which can be used for selection).

  For basic operators, the filter list must contain an operator, a field, and an optional value.
  For boolean operators, the filter list must contain a boolean operator, followed by a list of filters.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param f: Filter list.
  :type f: list
  :returns: Boolean Series corresponding to the filter.
  :rtype: pandas.Series
  :raises ValueError: If the operator is unknown.
  """
  operator = f[0]

  # check if operator is a boolean operator:
  if _is_bool_operator(operator):
    return resolve_bool_filter(df, f)
  else:
    field = f[1]
    if len(f) == 3:
      value = f[2]
    else:
      value = None

    if isinstance(value, str):
      if value.startswith("str:"):
        value = value[4:]

    if operator == ">": return df[field].fillna(0).gt(value)
    if operator == "<": return df[field].fillna(0).lt(value)
    if operator == ">=": return df[field].fillna(0).ge(value)
    if operator == "<=": return df[field].fillna(0).le(value)
    if operator == "==": return df[field].fillna(0).eq(value)
    if operator == "!=": return df[field].fillna(0).ne(value)
    if operator == "isin": return df[field].isin(value)
    if operator == "notin": return ~df[field].isin(value)
    if operator == "isempty": return pd.isna(df[field]) | df[field].astype(str).str.strip().eq("")
    if operator == "iszero": return df[field].eq(0)
    if operator == "contains":
      if isinstance(value, str):
        selection = df[field].str.contains(value)
      elif isinstance(value, list):
        selection = df[field].str.contains(value[0])
        for v in value[1:]: selection |= df[field].str.contains(v)
      else:
        raise ValueError(f"Value must be a string or list for operator {operator}, found: {type(value)}")
      return selection

  raise ValueError(f"Unknown operator {operator}")


def validate_filter_list(filters: list[list]):
  """
  Validate a list of filter lists.

  :param filters: List of filters (each filter is a list).
  :type filters: list[list]
  :returns: True if all filters are valid.
  :rtype: bool
  """
  for f in filters:
    validate_filter(f)
  return True


def validate_filter(f: list):
  """
  Validate a single filter list.

  Checks that the filter's operator is appropriate for the value type.

  :param f: Filter expressed as a list.
  :type f: list
  :returns: True if the filter is valid.
  :rtype: bool
  :raises ValueError: If the value type does not match the operator requirements.
  """
  operator = f[0]
  if operator in ["and", "or"]:
    pass
  else:
    value = f[2]

    if operator in [">", "<", ">=", "<="]:
      if not isinstance(value, (int, float, bool)):
        raise ValueError(f"Value must be a number for operator {operator}")
    if operator in ["isin", "notin"]:
      if not isinstance(value, list):
        raise ValueError(f"Value must be a list for operator {operator}")
    if operator == "contains":
      if not isinstance(value, str):
        raise ValueError(f"Value must be a string for operator {operator}")
  return True


def _is_basic_operator(s: str) -> bool:
  """
  Check if the operator is a basic comparison operator.

  :param s: Operator as a string.
  :type s: str
  :returns: True if it is a basic operator.
  :rtype: bool
  """
  return s in ["<", ">", "<=", ">=", "==", "!=", "isin", "notin", "contains"]


def _is_bool_operator(s: str) -> bool:
  """
  Check if the operator is a boolean operator.

  :param s: Operator as a string.
  :type s: str
  :returns: True if it is a boolean operator.
  :rtype: bool
  """
  return s in ["and", "or", "nand", "nor", "xor", "xnor"]
