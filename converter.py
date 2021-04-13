def converter(input, input_unit, output_unit):
    # Speed
    if (input_unit == "ms") & (output_unit == "knots"):
        return input * 1.94384
    elif (input_unit == "knots") & (output_unit == "ms"):
        return input / 1.94384
    # Altitude
    if (input_unit == "m") & (output_unit == "feet"):
        return input * 3.28084
    elif (input_unit == "feet") & (output_unit == "m"):
        return input / 3.28084
    # Distance
    if (input_unit == "m") & (output_unit == "nm"):
        return input * 0.000539957
    elif (input_unit == "nm") & (output_unit == "m"):
        return input / 0.000539957
