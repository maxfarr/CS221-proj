from rauth.service import OAuth1Service, OAuth1Session

class GoodreadsSession():
    """Handle OAuth sessions"""
    def __init__(self, client_key, client_secret, access_token=None,
                 access_token_secret=None):
        self.client_key = client_key
        self.client_secret = client_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret

    def oauth_init(self):
        """Start outh and return authorization url."""
        service = OAuth1Service(
            consumer_key=self.client_key,
            consumer_secret=self.client_secret,
            name='goodreads',
            request_token_url='https://www.goodreads.com/oauth/request_token',
            authorize_url='https://www.goodreads.com/oauth/authorize',
            access_token_url='https://www.goodreads.com/oauth/access_token',
            base_url='https://www.goodreads.com/'
        )
        request_token, request_token_secret = service.get_request_token(header_auth=True, method='POST')
        auth_url = service.get_authorize_url(request_token)
        # Store service for finalizing
        self.request_token = request_token
        self.request_token_secret = request_token_secret
        self.service = service
        return auth_url

    def oauth_finalize(self):
        """Once the user authorizes access, save access tokens"""
        self.session = self.service.get_auth_session(self.request_token,
                                                     self.request_token_secret)
        self.access_token = self.session.access_token
        self.access_token_secret = self.session.access_token_secret

    def oauth_resume(self):
        """Create session if access token and key are already available"""
        self.session = OAuth1Session(
            consumer_key=self.client_key,
            consumer_secret=self.client_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret)
    
    def authenticate(self):
        """Authenticate client to query requiring authorization"""
        if self.access_token and self.access_token_secret:
            self.oauth_resume()
        else:
            url = self.oauth_init()
            webbrowser.open(url)
            while input("Have you authorized me? (y/n)") != 'y':
                pass
            self.oauth_finalize()