from twilio.rest import Client

def make_outbound_call(account_sid, auth_token, from_number, to_number, person_name):
    client = Client(account_sid, auth_token)
    message = f"Hello, this is an automated message. The Driver Attention Monitoring System has detected that {person_name} is feeling drowsy. Please ensure that proper action is taken to address this issue and ensure safety. Thank you."

    call = client.calls.create(
        to=to_number,
        from_=from_number,
        twiml=f'<Response><Say>{message}</Say></Response>'
    )

    return call.sid

if __name__ == "__main__":  # placeholder
    # ACCOUNT_SID =
    # AUTH_TOKEN = 
    # FROM_NUMBER = 
    # TO_NUMBER = 
    # PERSON_NAME = 'John Snow'

    call_sid = make_outbound_call(ACCOUNT_SID, AUTH_TOKEN, FROM_NUMBER, TO_NUMBER, PERSON_NAME)
    print(f'Call initiated with SID: {call_sid}')
