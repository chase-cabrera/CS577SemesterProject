"""
Name Parser for FEC Contributor Data.
Parses names in "LASTNAME, FIRSTNAME MIDDLE SUFFIX" format.
"""


class NameParser:
    """
    Parse FEC contributor names in "LASTNAME, FIRSTNAME MIDDLE SUFFIX" format.
    """
    
    # Generational suffixes (12)
    GENERATIONAL = {
        'JR', 'JR.', 'SR', 'SR.', 
        'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'
    }
    
    # Professional/Academic suffixes (24)
    PROFESSIONAL = {
        # Medical
        'MD', 'M.D.', 'DO', 'D.O.', 'DDS', 'D.D.S.', 'DMD', 'DVM', 'RN', 'LPN', 'NP',
        # Academic
        'PHD', 'PH.D.', 'EDD', 'ED.D.', 'JD', 'J.D.', 'MBA', 'MPA', 'MS', 'MA',
        # Professional
        'ESQ', 'CPA', 'PE', 'RA', 'AIA', 'CFP', 'CLU', 'CPCU'
    }
    
    # All suffixes combined
    SUFFIXES = GENERATIONAL | PROFESSIONAL
    
    # Common prefixes to strip (helps with edge cases)
    PREFIXES = {
        'MR', 'MR.', 'MRS', 'MRS.', 'MS', 'MS.', 'MISS', 
        'DR', 'DR.', 'PROF', 'PROF.',
        'REV', 'REV.', 'FR', 'FR.', 'PASTOR',
        'HON', 'HON.', 'JUDGE',
        'GEN', 'COL', 'MAJ', 'CAPT', 'LT', 'SGT', 'CPL', 'PVT',
        'SIR', 'DAME'
    }
    
    def _is_prefix(self, token):
        """Check if a token is a prefix."""
        upper = token.upper()
        return upper in self.PREFIXES or upper.rstrip('.') in self.PREFIXES
    
    def _is_suffix(self, token):
        """Check if a token is a suffix."""
        upper = token.upper()
        return upper in self.SUFFIXES or upper.rstrip('.') in self.SUFFIXES
    
    def _strip_prefixes(self, tokens):
        """Remove prefix tokens from the beginning of the list."""
        while tokens and self._is_prefix(tokens[0]):
            tokens = tokens[1:]
        return tokens
    
    def _extract_suffixes(self, tokens):
        """Extract suffix tokens from the end of the list."""
        suffixes = []
        while tokens and self._is_suffix(tokens[-1]):
            suffixes.insert(0, tokens[-1])
            tokens = tokens[:-1]
        return tokens, ' '.join(suffixes) if suffixes else None
    
    def parse(self, name):
        """
        Parse a contributor name string into components.
        
        Args:
            name: Full name string (e.g., "SMITH, JOHN MICHAEL JR")
            
        Returns:
            Dictionary with first_name, last_name, middle_name, name_suffix
        """
        result = {
            'first_name': None,
            'last_name': None,
            'middle_name': None,
            'name_suffix': None
        }
        
        if not name or not isinstance(name, str):
            return result
        
        name = name.strip()
        if not name:
            return result
        
        # Split on comma FEC format is "LASTNAME, FIRSTNAME MIDDLE SUFFIX"
        if ',' in name:
            parts = name.split(',', 1)
            result['last_name'] = parts[0].strip()
            
            if len(parts) > 1 and parts[1].strip():
                remainder = parts[1].strip()
                tokens = remainder.split()
                
                if tokens:
                    # Strip prefixes (rare but possible: "DR JOHN")
                    tokens = self._strip_prefixes(tokens)
                    
                    if tokens:
                        # Extract suffixes from end
                        tokens, suffix = self._extract_suffixes(tokens)
                        result['name_suffix'] = suffix
                        
                        if tokens:
                            result['first_name'] = tokens[0]
                            
                            # Middle name is everything after first
                            if len(tokens) > 1:
                                result['middle_name'] = ' '.join(tokens[1:])
        else:
            # try to parse as "FIRSTNAME LASTNAME" or just "LASTNAME"
            tokens = name.split()
            
            # Strip prefixes
            tokens = self._strip_prefixes(tokens)
            
            if len(tokens) == 1:
                result['last_name'] = tokens[0]
            elif len(tokens) >= 2:
                # Extract suffixes from end
                tokens, suffix = self._extract_suffixes(tokens)
                result['name_suffix'] = suffix
                
                if len(tokens) >= 2:
                    result['first_name'] = tokens[0]
                    result['last_name'] = tokens[-1]
                    if len(tokens) > 2:
                        result['middle_name'] = ' '.join(tokens[1:-1])
                elif len(tokens) == 1:
                    result['last_name'] = tokens[0]
        
        return result
