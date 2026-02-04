# HEAD method invariant note

The HEAD-method assertion test (`test_validate_url_still_uses_head_method`) passed both before and after the User-Agent fix because `validate_url()` already enforced the `HEAD` method in the baseline implementation. This test is retained as a standard passing invariant check, so no failing artifact is expected or required.
