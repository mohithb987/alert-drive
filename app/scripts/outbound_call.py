from twilio.rest import Client

def make_outbound_call(account_sid='AC5bfd37a1c26fced23258153f6997041d', auth_token='93b7ad9458a0f181e9d5ab669b3394f1', from_number='+12513095329', to_number='+17162473107', person_name='John Snow'):
    client = Client(account_sid, auth_token)
    message = f"Hello, this is an automated message. The Driver Attention Monitoring System has detected that {person_name} is feeling drowsy. Please ensure that proper action is taken to address this issue and ensure safety. Thank you."

    call = client.calls.create(
        to=to_number,
        from_=from_number,
        twiml=f'<Response><Say>{message}</Say></Response>'
    )

    return call.sid

if __name__ == "__main__":
    ACCOUNT_SID = 'AC5bfd37a1c26fced23258153f6997041d'
    AUTH_TOKEN = '93b7ad9458a0f181e9d5ab669b3394f1'
    FROM_NUMBER = '+12513095329'
    TO_NUMBER = '+17162473107'
    PERSON_NAME = 'John Snow'

    call_sid = make_outbound_call(ACCOUNT_SID, AUTH_TOKEN, FROM_NUMBER, TO_NUMBER, PERSON_NAME)
    print(f'Call initiated with SID: {call_sid}')
