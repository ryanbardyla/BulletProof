// Test if string corruption is fixed
organism StringTest {
    fn birth() {
        express "=== STRING CORRUPTION FIX TEST ==="
        express "Line 1: This should be clear"
        express "Line 2: No corruption here"
        express "Line 3: Phoenix says hello"
        express "Line 4: Ember joins the chat"
        express "Line 5: Blaze is learning"
        express "Line 6: Spark completes the family"
        express ""
        express "If you can read all lines clearly..."
        express "THE STRING BUG IS FIXED!"
        express ""
        express "Testing longer strings now:"
        express "The quick brown fox jumps over the lazy dog"
        express "Pack my box with five dozen liquor jugs"
        express "How vexingly quick daft zebras jump"
        express ""
        express "Special characters: !@#$%^&*()"
        express "Numbers: 1234567890"
        express "Mixed: Test123 ABC xyz"
        express ""
        express "SUCCESS: All strings rendering correctly!"
    }
}