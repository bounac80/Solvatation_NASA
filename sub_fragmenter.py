# -*- coding: utf-8 -*-
''' Class for fragmenting molecules into molecular subgroups

MIT License

Copyright (C) 2019, Simon Mueller <simon.mueller@tuhh.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

#==============================================================================
#
#          SMILES --> SMARTS --> DECOMPOSITION INTO FUNCTIONAL GROUPS
#
#==============================================================================

class fragmenter:
    # tested with Python 3.8.8 and RDKit version 2021.09.4
    
    from rdkit import Chem
    import marshal as marshal
    from rdkit.Chem import rdmolops

    # does a substructure match and then checks whether the match 
    # is adjacent to previous matches
    @classmethod
    def get_substruct_matches(cls, mol_searched_for, mol_searched_in, atomIdxs_to_which_new_matches_have_to_be_adjacent):
        
        valid_matches = []
        
        if mol_searched_in.GetNumAtoms() >= mol_searched_for.GetNumAtoms():
            matches = mol_searched_in.GetSubstructMatches(mol_searched_for)
            
            if matches:
                for match in matches:
                        add_this_match = True
                        if len(atomIdxs_to_which_new_matches_have_to_be_adjacent) > 0:
                            add_this_match = False
                            
                            for i in match:
                                for neighbor in mol_searched_in.GetAtomWithIdx(i).GetNeighbors():
                                    if neighbor.GetIdx() in atomIdxs_to_which_new_matches_have_to_be_adjacent:
                                        add_this_match = True
                                        break
                                
                        if add_this_match:
                            valid_matches.append(match)
                
        return valid_matches
    
    # count heavier isotopes of hydrogen correctly
    @classmethod
    def get_heavy_atom_count(cls, mol):
        heavy_atom_count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                heavy_atom_count += 1
        
        return heavy_atom_count
    
    def __init__(self, fragmentation_scheme = {}, fragmentation_scheme_order = None, match_hydrogens = False, algorithm = '', n_atoms_cuttoff = -1, function_to_choose_fragmentation = False, n_max_fragmentations_to_find = -1):

        if not type(fragmentation_scheme) is dict:
            raise TypeError('fragmentation_scheme must be a dctionary with integers as keys and either strings or list of strings as values.')
            
        if len(fragmentation_scheme) == 0:
            raise ValueError('fragmentation_scheme must be provided.')
        
        if not algorithm in ['simple', 'complete', 'combined']:
            raise ValueError('Algorithm must be either simple ,complete or combined.')
            
        if algorithm == 'simple':
            if n_max_fragmentations_to_find != -1:
                raise ValueError('Setting n_max_fragmentations_to_find only makes sense with complete or combined algorithm.')
        
        self.algorithm = algorithm
        
        if algorithm in ['combined', 'complete']:
            if n_atoms_cuttoff == -1:
                raise ValueError('n_atoms_cuttoff needs to be specified for complete or combined algorithms.')
                
            if function_to_choose_fragmentation == False:
                raise ValueError('function_to_choose_fragmentation needs to be specified for complete or combined algorithms.')
                
            if not callable(function_to_choose_fragmentation):
                raise TypeError('function_to_choose_fragmentation needs to be a function.')
            else:
                if type(function_to_choose_fragmentation([{}, {}])) != dict:
                    raise TypeError('function_to_choose_fragmentation needs to take a list of fragmentations and choose one of it')
                
            if n_max_fragmentations_to_find != -1:
                if n_max_fragmentations_to_find < 1:
                    raise ValueError('n_max_fragmentations_to_find has to be 1 or higher.')

        if fragmentation_scheme_order is None:
            fragmentation_scheme_order = []

        if algorithm in ['simple', 'combined']:
            assert len(fragmentation_scheme) == len(fragmentation_scheme_order)
        else:
            fragmentation_scheme_order = [key for key in fragmentation_scheme.keys()]
            
        self.n_max_fragmentations_to_find = n_max_fragmentations_to_find
        
        self.n_atoms_cuttoff = n_atoms_cuttoff

        self.match_hydrogens = match_hydrogens
        
        self.fragmentation_scheme = fragmentation_scheme
        
        self.function_to_choose_fragmentation = function_to_choose_fragmentation
        
        # create a lookup dictionaries to faster finding a group number
        self._fragmentation_scheme_group_number_lookup = {}
        self._fragmentation_scheme_pattern_lookup = {}
        self.fragmentation_scheme_order = fragmentation_scheme_order

        for group_number, list_SMARTS in fragmentation_scheme.items():
            
            if type(list_SMARTS) is not list:
                list_SMARTS = [list_SMARTS]
                
            for SMARTS in list_SMARTS:
                if SMARTS != '':
                    self._fragmentation_scheme_group_number_lookup[SMARTS] = group_number
                    
                    mol_SMARTS = fragmenter.Chem.MolFromSmarts(SMARTS)
                    self._fragmentation_scheme_pattern_lookup[SMARTS] = mol_SMARTS

    def fragment(self, SMILES_or_molecule):
        
        if type(SMILES_or_molecule) is str:
            mol_SMILES = fragmenter.Chem.MolFromSmiles(SMILES_or_molecule)
            mol_SMILES = fragmenter.Chem.AddHs(mol_SMILES) if self.match_hydrogens else mol_SMILES
            is_valid_SMILES = mol_SMILES is not None
            
            if not is_valid_SMILES:
                raise ValueError('Following SMILES is not valid: ' + SMILES_or_molecule)

        else:
            mol_SMILES = SMILES_or_molecule

        # iterate over all separated molecules
        success = []
        fragmentation = {}
        fragmentation_matches = {}
        for mol in fragmenter.rdmolops.GetMolFrags(mol_SMILES, asMols = True):

            this_mol_fragmentation, this_mol_success = self.__get_fragmentation(mol)
    
            for SMARTS, matches in this_mol_fragmentation.items():
                group_number = self._fragmentation_scheme_group_number_lookup[SMARTS]
                
                if not group_number in fragmentation:
                    fragmentation[group_number] = 0
                    fragmentation_matches[group_number] = []
                
                fragmentation[group_number] += len(matches)
                fragmentation_matches[group_number].extend(matches)   
                
            success.append(this_mol_success)

        return fragmentation, all(success), fragmentation_matches
    
    def fragment_complete(self, SMILES_or_molecule):
        
        if type(SMILES_or_molecule) is str:
            mol_SMILES = fragmenter.Chem.MolFromSmiles(SMILES_or_molecule)
            mol_SMILES = fragmenter.Chem.AddHs(mol_SMILES) if self.match_hydrogens else mol_SMILES
            is_valid_SMILES = mol_SMILES is not None
            
            if not is_valid_SMILES:
                raise ValueError('Following SMILES is not valid: ' + SMILES_or_molecule)

        else:
            mol_SMILES = SMILES_or_molecule

        if len(fragmenter.rdmolops.GetMolFrags(mol_SMILES)) != 1:
            raise ValueError('fragment_complete does not accept multifragment molecules.')

        temp_fragmentations, success = self.__complete_fragmentation(mol_SMILES)

        fragmentations = []
        fragmentations_matches = []
        for temp_fragmentation in temp_fragmentations:
            fragmentation = {}
            fragmentation_matches = {}
            for SMARTS, matches in temp_fragmentation.items():
                group_number = self._fragmentation_scheme_group_number_lookup[SMARTS]

                fragmentation[group_number] = len(matches)
                fragmentation_matches[group_number] = matches

            fragmentations.append(fragmentation)
            fragmentations_matches.append(fragmentation_matches)

        return fragmentations, success, fragmentations_matches


    def __get_fragmentation(self, mol_SMILES):
        
        success = False
        fragmentation = {}
        if self.algorithm in ['simple', 'combined']:
            fragmentation, success = self.__simple_fragmentation(mol_SMILES)
        
        if success:
            return fragmentation, success
        
        if self.algorithm in ['combined', 'complete']:
            fragmentations, success = self.__complete_fragmentation(mol_SMILES)
            
            if success:
                fragmentation = self.function_to_choose_fragmentation(fragmentations)
        
        return fragmentation, success
    
    def __simple_fragmentation(self, mol_SMILES):

        if self.match_hydrogens:
            target_atom_count = len(mol_SMILES.GetAtoms())
        else:
            target_atom_count = fragmenter.get_heavy_atom_count(mol_SMILES)
    
        success = False
        fragmentation = {}
        
        fragmentation, atomIdxs_included_in_fragmentation = self.__search_non_overlapping_solution(mol_SMILES, {}, set(), set())
        success = len(atomIdxs_included_in_fragmentation) == target_atom_count
        
        # if not successful, clean up molecule and search again
        level = 1
        while not success:
            fragmentation_so_far , atomIdxs_included_in_fragmentation_so_far = fragmenter.__clean_molecule_surrounding_unmatched_atoms(mol_SMILES, fragmentation, atomIdxs_included_in_fragmentation, level)
            level += 1
            
            if len(atomIdxs_included_in_fragmentation_so_far) == 0:
                break
            
            fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far = self.__search_non_overlapping_solution(mol_SMILES, fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far)
            
            success = len(atomIdxs_included_in_fragmentation_so_far) == target_atom_count
            
            if success:
                fragmentation = fragmentation_so_far
            
        return fragmentation, success
    
    def __search_non_overlapping_solution(self, mol_searched_in, fragmentation, atomIdxs_included_in_fragmentation, atomIdxs_to_which_new_matches_have_to_be_adjacent):
        
        n_atomIdxs_included_in_fragmentation = len(atomIdxs_included_in_fragmentation) - 1
        
        while n_atomIdxs_included_in_fragmentation != len(atomIdxs_included_in_fragmentation):
            n_atomIdxs_included_in_fragmentation = len(atomIdxs_included_in_fragmentation)
            
             
            for group_number in self.fragmentation_scheme_order:
                list_SMARTS = self.fragmentation_scheme[group_number]
                if type(list_SMARTS) is not list:
                    list_SMARTS = [list_SMARTS]
                
                for SMARTS in list_SMARTS:
                    if SMARTS != "":  
                        fragmentation, atomIdxs_included_in_fragmentation = self.__get_next_non_overlapping_match(mol_searched_in, SMARTS, fragmentation, atomIdxs_included_in_fragmentation, atomIdxs_to_which_new_matches_have_to_be_adjacent)

        return fragmentation, atomIdxs_included_in_fragmentation

    def __get_next_non_overlapping_match(self, mol_searched_in, SMARTS, fragmentation, atomIdxs_included_in_fragmentation, atomIdxs_to_which_new_matches_have_to_be_adjacent):
        
        mol_searched_for = self._fragmentation_scheme_pattern_lookup[SMARTS]
        
        if atomIdxs_to_which_new_matches_have_to_be_adjacent:
            matches = fragmenter.get_substruct_matches(mol_searched_for, mol_searched_in, atomIdxs_to_which_new_matches_have_to_be_adjacent)
        else:
            matches = fragmenter.get_substruct_matches(mol_searched_for, mol_searched_in, set())
        
        if matches:
            for match in matches:
                all_atoms_of_new_match_are_unassigned = atomIdxs_included_in_fragmentation.isdisjoint(match)
        
                if all_atoms_of_new_match_are_unassigned:
                    if not SMARTS in fragmentation:
                        fragmentation[SMARTS] = []
                        
                    fragmentation[SMARTS].append(match)
                    atomIdxs_included_in_fragmentation.update(match)   
                
        return fragmentation, atomIdxs_included_in_fragmentation

    @classmethod
    def __clean_molecule_surrounding_unmatched_atoms(cls, mol_searched_in, fragmentation, atomIdxs_included_in_fragmentation, level):
    
        for i in range(0, level):
            
            atoms_missing = set(range(0, fragmenter.get_heavy_atom_count(mol_searched_in))).difference(atomIdxs_included_in_fragmentation)
                        
            new_fragmentation = fragmenter.marshal.loads(fragmenter.marshal.dumps(fragmentation))
            
            for atomIdx in atoms_missing:
                for neighbor in mol_searched_in.GetAtomWithIdx(atomIdx).GetNeighbors():
                    for smart, atoms_found in fragmentation.items():
                        for atoms in atoms_found:
                            if neighbor.GetIdx() in atoms:
                                if smart in new_fragmentation:
                                    if new_fragmentation[smart].count(atoms) > 0:
                                        new_fragmentation[smart].remove(atoms)
                                
                        if smart in new_fragmentation:
                            if len(new_fragmentation[smart]) == 0:
                                new_fragmentation.pop(smart)
                                
                                
            new_atomIdxs_included_in_fragmentation = set()
            for i in new_fragmentation.values():
                for j in i:
                    new_atomIdxs_included_in_fragmentation.update(j)
                    
            atomIdxs_included_in_fragmentation = new_atomIdxs_included_in_fragmentation
            fragmentation = new_fragmentation
            
        return fragmentation, atomIdxs_included_in_fragmentation
       
    def __complete_fragmentation(self, mol_SMILES):
    
        heavy_atom_count = fragmenter.get_heavy_atom_count(mol_SMILES)
        
        if heavy_atom_count > self.n_atoms_cuttoff:
            return {}, False
        
        completed_fragmentations = []
        groups_leading_to_incomplete_fragmentations = []        
        completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found = self.__get_next_non_overlapping_adjacent_match_recursively(mol_SMILES, heavy_atom_count, completed_fragmentations, groups_leading_to_incomplete_fragmentations, {}, set(), set(), self.n_max_fragmentations_to_find)
        success = len(completed_fragmentations) > 0
        
        return completed_fragmentations, success
        
    def __get_next_non_overlapping_adjacent_match_recursively(self, mol_searched_in, heavy_atom_count, completed_fragmentations, groups_leading_to_incomplete_fragmentations, fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far, atomIdxs_to_which_new_matches_have_to_be_adjacent, n_max_fragmentations_to_find = -1):
      
        n_completed_fragmentations = len(completed_fragmentations)
        incomplete_fragmentation_found = False
        complete_fragmentation_found = False
        
        if len(completed_fragmentations) == n_max_fragmentations_to_find:
            return completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found
                    
                
        for group_number in self.fragmentation_scheme_order:
            list_SMARTS = self.fragmentation_scheme[group_number]
            
            if complete_fragmentation_found:
                break
            
            if type(list_SMARTS) is not list:
                list_SMARTS = [list_SMARTS]
            
            for SMARTS in list_SMARTS:
                if complete_fragmentation_found:
                    break
                
                if SMARTS != "":  
                    matches = fragmenter.get_substruct_matches(self._fragmentation_scheme_pattern_lookup[SMARTS], mol_searched_in, atomIdxs_included_in_fragmentation_so_far)
                    
                    for match in matches:
                        
                        # only allow non-overlapping matches
                        all_atoms_are_unassigned = atomIdxs_included_in_fragmentation_so_far.isdisjoint(match)
                        if not all_atoms_are_unassigned:
                            continue
                        
                        # only allow matches that do not contain groups leading to incomplete matches
                        for groups_leading_to_incomplete_fragmentation in groups_leading_to_incomplete_fragmentations:
                            if fragmenter.__is_fragmentation_subset_of_other_fragmentation(groups_leading_to_incomplete_fragmentation, fragmentation_so_far):
                                return completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found
                        
                        # only allow matches that will lead to new fragmentations
                        use_this_match = True
                        n_found_groups = len(fragmentation_so_far)
                        
                        for completed_fragmentation in completed_fragmentations:
                            
                            if not SMARTS in completed_fragmentation:
                                continue
                            
                            if n_found_groups == 0:
                                use_this_match = not fragmenter.__is_match_contained_in_fragmentation(match, SMARTS, completed_fragmentation)
                            else:
                                if fragmenter.__is_fragmentation_subset_of_other_fragmentation(fragmentation_so_far, completed_fragmentation):
                                    use_this_match = not fragmenter.__is_match_contained_in_fragmentation(match, SMARTS, completed_fragmentation)
                            
                            if not use_this_match:
                                break
                                
                        if not use_this_match:
                            continue
                        
                        # make a deepcopy here, otherwise the variables are modified down the road
                        # marshal is used here because it works faster than copy.deepcopy
                        this_SMARTS_fragmentation_so_far = fragmenter.marshal.loads(fragmenter.marshal.dumps(fragmentation_so_far))
                        this_SMARTS_atomIdxs_included_in_fragmentation_so_far = atomIdxs_included_in_fragmentation_so_far.copy()
                        
                        if not SMARTS in this_SMARTS_fragmentation_so_far:
                            this_SMARTS_fragmentation_so_far[SMARTS] = []
                            
                        this_SMARTS_fragmentation_so_far[SMARTS].append(match)
                        this_SMARTS_atomIdxs_included_in_fragmentation_so_far.update(match)
                        
                        # only allow matches that do not contain groups leading to incomplete matches
                        for groups_leading_to_incomplete_match in groups_leading_to_incomplete_fragmentations:
                            if fragmenter.__is_fragmentation_subset_of_other_fragmentation(groups_leading_to_incomplete_match, this_SMARTS_fragmentation_so_far):
                                use_this_match = False
                                break
                            
                        if not use_this_match:
                            continue
                        
                        # if the complete molecule has not been fragmented, continue to do so
                        if len(this_SMARTS_atomIdxs_included_in_fragmentation_so_far) < heavy_atom_count:
                            completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found = self.__get_next_non_overlapping_adjacent_match_recursively(mol_searched_in, heavy_atom_count, completed_fragmentations, groups_leading_to_incomplete_fragmentations, this_SMARTS_fragmentation_so_far, this_SMARTS_atomIdxs_included_in_fragmentation_so_far, this_SMARTS_atomIdxs_included_in_fragmentation_so_far, n_max_fragmentations_to_find)
                            break
                        
                        # if the complete molecule has been fragmented, save and return
                        if len(this_SMARTS_atomIdxs_included_in_fragmentation_so_far) == heavy_atom_count:                 
                            completed_fragmentations.append(this_SMARTS_fragmentation_so_far)
                            complete_fragmentation_found = True
                            break
                        
        # if until here no new fragmentation was found check whether an incomplete fragmentation was found
        if n_completed_fragmentations == len(completed_fragmentations):      
            
            if not incomplete_fragmentation_found:
                
                incomplete_matched_groups = {}
                
                if len(atomIdxs_included_in_fragmentation_so_far) > 0:
                    unassignes_atom_idx = set(range(0, heavy_atom_count)).difference(atomIdxs_included_in_fragmentation_so_far)
                    for atom_idx in unassignes_atom_idx:
                        neighbor_atoms_idx = [i.GetIdx() for i in mol_searched_in.GetAtomWithIdx(atom_idx).GetNeighbors()]
                        
                        for neighbor_atom_idx in neighbor_atoms_idx:
                            for found_smarts, found_matches in fragmentation_so_far.items():
                                for found_match in found_matches:
                                    if neighbor_atom_idx in found_match:
                                        if not found_smarts in incomplete_matched_groups:
                                            incomplete_matched_groups[found_smarts] = []
                                            
                                        if found_match not in incomplete_matched_groups[found_smarts]:
                                            incomplete_matched_groups[found_smarts].append(found_match)
                                    
                    is_subset_of_groups_already_found = False
                    indexes_to_remove = []
                    
                    for idx, groups_leading_to_incomplete_match in enumerate(groups_leading_to_incomplete_fragmentations):
                        is_subset_of_groups_already_found = fragmenter.__is_fragmentation_subset_of_other_fragmentation(incomplete_matched_groups, groups_leading_to_incomplete_match)
                        if is_subset_of_groups_already_found:
                            indexes_to_remove.append(idx)
                        
                    for index in sorted(indexes_to_remove, reverse=True):
                        del groups_leading_to_incomplete_fragmentations[index]
                        
                    groups_leading_to_incomplete_fragmentations.append(incomplete_matched_groups)
                    groups_leading_to_incomplete_fragmentations = sorted(groups_leading_to_incomplete_fragmentations, key = len)
                    
                    incomplete_fragmentation_found =  True
    
        return completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found
    
    @classmethod
    def __is_fragmentation_subset_of_other_fragmentation(cls, fragmentation, other_fragmentation):
        n_found_groups = len(fragmentation)
        n_found_other_groups = len(other_fragmentation)
        
        if n_found_groups == 0:
            return False
            
        if n_found_other_groups < n_found_groups:
            return False
        
        n_found_SMARTS_that_are_subset = 0
        for found_SMARTS, _ in fragmentation.items():
            if found_SMARTS in other_fragmentation:
                found_matches_set = set(frozenset(i) for i in fragmentation[found_SMARTS])
                found_other_matches_set =  set(frozenset(i) for i in other_fragmentation[found_SMARTS])
                
                if found_matches_set.issubset(found_other_matches_set):
                    n_found_SMARTS_that_are_subset += 1
            else:
                return False
            
        return n_found_SMARTS_that_are_subset == n_found_groups
    
    @classmethod
    def __is_match_contained_in_fragmentation(cls, match, SMARTS, fragmentation):
        if not SMARTS in fragmentation:
            return False
            
        found_matches_set = set(frozenset(i) for i in fragmentation[SMARTS])
        match_set = set(match)
        
        return match_set in found_matches_set

#==============================================================================
#
#                     DECOMPOSITION FROM SMILES INPUT
#
#==============================================================================

import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import NumRadicalElectrons
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

def get_radical_atoms(smi):
    mol = Chem.MolFromSmiles(smi)
    radical_atoms = []
    if mol:
        # Get the radical electrons for each atom
        for atom in mol.GetAtoms():
            # Checking if an atom has an odd number of electrons (radical site)
            if atom.GetNumRadicalElectrons() > 0:
                radical_atoms.append(atom.GetIdx())  # Add the atom index (or any other atom property)
    return radical_atoms

def has_oxygen_radical(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "O" and atom.GetNumRadicalElectrons() > 0:
                return True  # Found an oxygen radical
    return False  # No oxygen radicals found

def is_radical(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        num_radicals = NumRadicalElectrons(mol)
        if num_radicals > 0:
            return True, num_radicals
    return False, 0

def sum_digits_string(str1):
    sum_tot_grp = 0
    num_type_grp = 0
    i = 0
    for x in str1:
        i = i + 1
        if x == ":":
            num_type_grp = num_type_grp + 1
        if x.isdigit() == True and str1[i-3] == ":" :
            z = int(x)
            sum_tot_grp = sum_tot_grp + z
    return [sum_tot_grp,num_type_grp]

def groupocc(groups_string, groups_UNIFAC):
    # Initialize the occurrence vector
    Ngr = len(groups_UNIFAC)
    occ = np.zeros((Ngr,1))
    # Add a comma at the end to make group extraction simpler
    groups_string += ','
    sub1 = ''
    for i, group in enumerate(groups_UNIFAC):
        group = str(group)
        if group in groups_string:
            sub1 = "'" + group + "'" + ':'
            sub2 = ','
            # getting index of substrings
            idx1 = groups_string.index(sub1)
            idx2 = groups_string.index(sub2,idx1)
            # getting elements in between
            res = ''
            for idx in range(idx1 + len(sub1) + 1, idx2):
                res = res + groups_string[idx]
            occ[i] = int(res)
            occ = occ.reshape((len(occ), 1))
    return occ
    
def bestfragmentation(smi):
    fragmentation_scheme = {
        # UNIFAC groups
        '[CH3]' : '[CH3]',
        '[CH2]' : '[CH2]', 
        '[CH]' : '[CH1]', 
        '[C]' : '[CH0]',
        '[CH2=CH]' : '[CH2]=[CH]',
        '[CH=CH]' : '[CH]=[CH]',
        '[CH2=C]' : '[CH2]=[C]',
        '[CH=C]' : '[CH]=[CH0]',
        '[C=C]' : '[CH0]=[CH0]',
        '[ACH]' : '[cH]',
        '[AC]' : '[cH0]',
        '[ACCH3]' : '[c][CH3]',
        '[ACCH2]' : '[c][CH2]',
        '[ACCH]' : '[c][CH]',
        '[OH]' : '[OH]',
        '[CH3OH]' : '[CH3][OH]',
        '[H2O]' : '[OH2]',
        '[ACOH]' : '[c][OH]',
        '[CH3CO]' : '[CH3][CH0]=O',
        '[CH2CO]' : '[CH2][CH0]=O',
        '[CH=O]' : '[CH]=O',
        '[CH3COO]' : '[CH3]C(=O)[OH0]',
        '[CH2COO]' : '[CH2]C(=O)[OH0]',
        '[HCOO]' : '[CH](=O)[OH0]',
        '[CH3O]' : '[CH3][OH0]',
        '[CH2O]' : '[CH2][OH0]',
        '[CHO]' : '[CH][OH0]',
        '[THF]' : '[CH2;R][OH0]',
        '[CH3NH2]' : '[CH3][NH2]',
        '[CH2NH2]' : '[CH2][NH2]',
        '[CHNH2]' : '[CH][NH2]',
        '[CH3NH]' : '[CH3][NH]',
        '[CH2NH]' : '[CH2][NH]',
        '[CHNH]' : '[CH][NH]',
        '[CH3N]' : '[CH3][N]',
        '[CH2N]' : '[CH2][N]',
        '[ACNH2]' : '[c][NH2]',
        '[C5H5N]' : 'n1[cH][cH][cH][cH][cH]1',
        '[C5H4N]' : 'n1[c][cH][cH][cH][cH]1',
        '[C5H3N]' : 'n1[c][c][cH][cH][cH]1',
        '[CH3CN]' : '[CH3]C#N',
        '[CH2CN]' : '[CH2]C#N',
        '[COOH]' : '[CH0](=O)[OH]',
        '[HCOOH]' : '[CH](=O)[OH]',
        '[CH2Cl]' : '[CH2]Cl',
        '[CHCl]' : '[CH]Cl',
        '[CCl]' : '[CH0]Cl',
        '[CH2Cl2]' : '[CH2](Cl)Cl',
        '[CHCl2]' : '[CH](Cl)Cl',
        '[CCl2]' : 'C(Cl)Cl',
        '[CHCl3]' : '[CH](Cl)(Cl)Cl',
        '[CCl3]' : 'C(Cl)(Cl)(Cl)',
        '[CCl4]' : 'C(Cl)(Cl)(Cl)(Cl)',
        '[ACCl]' : '[c]Cl',
        '[CH3NO2]' : '[CH3][N+](=O)[O-]',
        '[CH2NO2]' : '[CH2][N+](=O)[O-]',
        '[CHNO2]' : '[CH][N+](=O)[O-]',
        '[ACNO2]' : '[c][N+](=O)[O-]',
        '[CS2]' : 'C(=S)=S',
        '[CH3SH]' : '[CH3][SH]',
        '[CH2SH]' : '[CH2][SH]',
        '[FURFURAL]' : 'O=[CH]c1[cH][cH][cH]o1',
        '[DOH]' : '[OH][CH2][CH2][OH]',
        '[I]' : '[IH0]',
        '[BR]' : '[BrH0]',
        '[CH=-C]' : '[CH]#C',
        '[C=-C]' : 'C#C',
        '[DMS]O' : '[CH3]S(=O)[CH3]',
        '[ACRY]' : '[CH2]=[CH1][C]#N',
        '[Cl-(C=C)' : '[$(Cl[C]=[C])]',
        '[ACF]' : '[c]F',
        '[DMF]' : '[CH](=O)N([CH3])[CH3]',
        '[CF3]' : 'C(F)(F)F',
        '[CF2]' : 'C(F)F',
        '[CF]' : '[C]F',
        '[COO]' : '[CH0](=O)[OH0]',
        '[SiH3]' : '[SiH3]',
        '[SiH2]' : '[SiH2]',
        '[SiH]' : '[SiH]',
        '[Si]' : '[Si]',
        '[SiO]' : '[Si][OH0]',
        '[NMP]' : '[CH3]N1[CH2][CH2][CH2]C(=O)1',
        '[CCl3F]' : 'C(Cl)(Cl)(Cl)F',
        '[CCl2F]' : 'C(Cl)(Cl)F',
        '[HCCl2F]' : '[CH](Cl)(Cl)F',
        '[HCClF]' : '[CH](Cl)F',
        '[CClF2]' : 'C(Cl)(F)F',
        '[HCClF2]' : '[CH](Cl)(F)F',
        '[CClF]' : 'C(Cl)(F)(F)F',
        '[CCl2F2]' : 'C(Cl)(Cl)(F)F',
        '[AMH2]' : 'C(=O)[NH2]',
        '[AMHCH3]' : 'C(=O)[NH][CH3]',
        '[AMHCH2]' : 'C(=O)[NH][CH2]',
        '[AM(CH3)2]' : 'C(=O)N([CH3])[CH3]',
        '[AMCH3CH2]' : 'C(=O)N([CH3])[CH2]',
        '[AM(CH2)2]' : 'C(=O)N([CH2])[CH2]',
        '[C2H5O2]' : '[OH0;!$(OC=O);!R][CH2;!R][CH2;!R][OH]',
        '[C2H4O2]' : '[OH0;!$(OC=O);!R][CH;!R][CH2;!R][OH]',
        '[CH3S]' : '[CH3]S',
        '[CH2S]' : '[CH2]S',
        '[CHS]' : '[CH]S',
        '[MORPH]' : '[CH2]1[CH2][NH][CH2][CH2]O1',
        '[C4H4S]' : '[cH]1[cH][s;X2][cH][cH]1',
        '[C4H3S]' : '[c]1[cH][s;X2][cH][cH]1',
        '[C4H2S]' : '[c]1[c][s;X2][cH][cH]1',
        '[H2]' : '[H][H]',
        '[O2]' : '[O]=[O]',
        '[N2]' : '[N2]',
        '[CO]' : '[C-]#[O]',
        '[CO2]' : 'O=C=O',
        '[H2S]' : '[H2S]',
        '[CH4]' : '[CH4]',
        '[C2H2]': '[CH]#[CH]',
        '[C2H4]': '[CH2]=[CH2]',
        '[C2H6]': '[CH3][CH3]',
        '[C3H6]': '[CH3][CH]=[CH2]',
        '[C3H8]': '[CH3][CH2][CH3]',
        '[C3H4]': '[CH3]C#C',
        '[C4H10]': '[CH3][CH2][CH2][CH3]',
        '[Ar]' : '[Ar]',
        '[Cl2]' : '[Cl2]',
        '[NH3]' : '[NH3]',
        '[SO2]' : '[SO2]',
        '[C(=O)OOH]' : 'C=O[OH0][OH]',
        '[CH2OOH]' : '[CH2][OH0][OH]',
        '[C-O-OH]' : '[CH0][OH0][OH]',
        '[CHOOH]' : '[CH][OH0][OH]',
        '[CHOOCH2]' : '[CH][OH0][OH0][CH]',
        '[CH3OOC]' : '[CH3][OH0][OH0][CH]',
        '[CH2ONO2]' : '[CH2][OH0][N+](=O)[O-]',
        '[CHONO2]' : '[CH][OH0][N+](=O)[O-]',
        '[HCH=O]' : '[CH2]=O',
        '[CH3OOH]' : '[CH3][OH0][OH]',
        '[CH2=C=O]' : '[CH2]=[CH0]=[OH0]',
        '[H2O2]' : '[OH][OH]',
        # new groups
        '[C2H5OH]' : '[CH3][CH2][OH]',
        '[CH3C=OCH3]' : '[CH3]C(=O)[CH3]',
        '[CH3COOH]' : '[CH3][OH0][OH]',
        '[CH3CH=O]' : '[CH3][CH][OH0]',
        # free radicals (hydrogen atom)
        '[H(.)]' : '[H]*',
        # free radicals (acetylenic)
        '[C(.)=-C]' : 'C#[CH0]',
        '[C2H(.)]' : '[CH]#[CH0]', 
        # free radicals (alkoxyl)
        '[O(.)]' : '[O]*',
        '[HO(.)]' : '[OH2;H1]',
        '[CH3O(.)]' : '[CH3][#8;X2;v1;H0]',
        '[C2H5O(.)]' : '[CH3][CH2][OH0]',
        '[ACO(.)]' : '[c][OH0]',
        '[ACO(.)-phenol]' :'ccccc[c][OH0]', # *** to check ***
        # free radicals (benzyl)
        '[ACCH2(.)]' : '[c][CH2]', # *** to check ***
        '[ACCH(.)]' :  '[c][CH]',
        # free radicals (carbonyl)
        '[C(.)=O]' : '[#6;X2;v3;H0]=O', 
        '[COO(.)]' : '[CH](=O)[OH0]',
        '[(.)COO]' : '[#6;X2;v3;H0](=O)O',
        '[HC(.)=O]' : '[#6;X2;v3;H1]=O', 
        '[CH3C(.)=O]' : '[CH3][CH0]=O', 
        '[CH3COO(.)]' : '[CH3][C](=O)[OH0]', 
        '[HCOO(.)]' : '[CH1](=O)[OH0]',
        '[(.)COOH]' : '[CH0](=O)[OH]',
        # free radicals (peroxyl)
        '[H(.)O2]' : '[OH][OH0]',
        '[C(=O)OO(.)]' : 'C=O[OH0][OH0]',
        '[CHOO(.)]' : '[CH][OH0][OH0]',
        '[CH3OO(.)]' : '[CH3][OH0][OH0]',
        '[C2H5OO(.)]' : '[CH3][CH2][OH0][OH0]',
        '[CH2OO(.)]' : '[CH2][OH0][OH0]',
        '[C-O-O(.)]' : '[CH0][OH0][OH0]',
        # free radicals (phenyl)
        '[AC(.)]' : '[cH0]',
        # free radicals (primary)
        '[CH2(.)]' : '[#6;X3v3+0;H2]',
        '[C4H9(.)-primary]' : '[CH3][CH2][CH2][#6;X3v3+0;H2]', #butane primary
        '[C2H5(.)]' : '[CH3][#6;X3v3+0;H2]', #ethane primary
        '[CH3(.)]' : '[#6;X3v3+0;H3]', # methane
        '[CH2(.)OH]' : '[#6;X3v3+0;H2][OH]', # methanol primary
        '[C3H7(.)-primary]' : '[CH3][CH2][#6;X3v3+0;H2]', #propane primary
        '[C3H5(.)-primary]' : '[#6;X3v3+0;H2][CH]=[CH2]', #propene primary
        # free radicals (secondary)
        '[CH(.)]' : '[#6;X3v3+0;H1]',
        '[C4H9(.)-secondary]' : '[CH3][CH2][#6;X3v3+0;H1][CH3]', #butane secondary
        '[CH(.)-resonance]' : '[#6;X3v3+0;H1][CH0]=O', 
        '[C3H7(.)-secondary]' : '[CH3][#6;X3v3+0;H1][CH3]', #propane secondary
        # free radicals (tentiary)
        '[C(.)]' : '[#6;X3v3+0;H0]',
        # free radicals (vinyl)
        '[C(.)=C]' : '[CH2]=[CH]',
        '[CH(.)=CH]' : '[#6;X3v3+0;H1]=[CH]',
        '[C2H3(.)-vinyl]' : '[CH3][CH]=[#6;X3v3+0;H1]',
        '[CH(.)CCO]' : '[#6;X2v3+0;H1]=[CH0]=[OH0]',
        '[C(.)=CH]' : '[#6;X3v3+0;H0]=[CH]',
        '[CH2=C(.)]' : '[CH2]=[#6;X3v3+0;H0]',
        '[C3H5(.)-vinyl]' : '[CH3][#6;X3v3+0;H1]=[CH2]'
        }
    
    radical_status, num_radicals = is_radical(smi)

    if not radical_status:
        # Remove radical-containing fragments
        filtered_scheme = {k: v for k, v in fragmentation_scheme.items() if "(.)" not in k}
    else:
        filtered_scheme = fragmentation_scheme
    
    
    frg = fragmenter(filtered_scheme, algorithm='complete', n_atoms_cuttoff=50, function_to_choose_fragmentation=lambda x: x[0])
    skip = 0
    if type(smi) is str:
        mol_SMILES = fragmenter.Chem.MolFromSmiles(smi)
        is_valid_SMILES = mol_SMILES is not None
        if not is_valid_SMILES:
            skip = 1
            final_fragmentation = "[]"       
    else:
        mol_SMILES = smi
    if skip == 0:
        # groups that are entire molecules or having problems with the SMARTS notation
        if smi == '[HH]':
            final_fragmentation = "'[H2]': 1"
            fragmentations = "[{'[H2]': 1]}"
        elif smi == '[H]': 
            final_fragmentation = "'[H(.)]': 1"
            fragmentations = "[{'[H(.)]': 1]}"
            # acetylenic
        elif smi == 'C#[C]' or smi == '[C]#C':
            final_fragmentation = "'[C2H(.)]': 1"
            fragmentations = "[{'[C2H(.)]: 1]}"
        elif smi == 'CC#[C]' or smi == '[C]#CC':
                final_fragmentation = "'[C(.)=-C]': 1, '[CH3]': 1"
                fragmentations = "[{'[C(.)=-C]': 1, '[CH3]': 1]}"
            # alkoxy
        elif smi == '[O]': 
            final_fragmentation = "'[HO(.)]': 1"
            fragmentations = "[{'[HO(.)]': 1]}"
        elif smi == 'C[O]' or smi == '[O]C': 
            final_fragmentation = "'[CH3O(.)]': 1"
            fragmentations = "[{'[CH3O(.)]': 1]}"
            # Peroxyl
        elif smi == 'O[O]' or smi == '[O]O': 
            final_fragmentation = "'[H(.)O2]': 1"
            fragmentations = "[{'[H(.)O2]': 1]}"
            # Carbonyl
        elif smi == '[CH](=O)': 
            final_fragmentation = "'[HC(.)=O]': 1"
            fragmentations = "[{'[HC(.)=O]': 1]}"
        elif smi == 'CC(=O)[O]': 
            final_fragmentation = "'[CH3COO(.)]': 1"
            fragmentations = "[{'[CH3COO(.)]': 1]}"
        elif smi == '[CH3]': 
            final_fragmentation = "'[CH3(.)]': 1"
            fragmentations = "[{[CH3(.)]': 1]}"
        elif smi == '[CH2]O' or smi == 'O[CH2]': 
            final_fragmentation = "'[CH2(.)OH]': 1"
            fragmentations = "[{'[CH2(.)OH]': 1]}"
        elif smi == '[C](=O)C':
            final_fragmentation = "'[CH3C(.)=O]': 1"
            fragmentations = "[{'[CH3C(.)=O]': 1]}"
        else:
            fragmentations, success, fragmentations_matches = frg.fragment_complete(smi)
            # all fragmentations found
            fragmentations_str_full = str(fragmentations) # all fragmentations
            fragmentations_str_full = fragmentations_str_full.replace('[{', '')
            fragmentations_str_full = fragmentations_str_full.replace('}]', '')
            fragmentations_str_split = fragmentations_str_full.split("}, {")
            # final fragmentation is the one with the smallest number of groups (valid for closed-shell molecules)
            final_fragmentation = '[]' #fragmentations_str_split[0]
            number_groups_final = 8888
            number_type_groups_final = 8888
            # [number_groups_final,number_type_groups_final] = sum_digits_string(final_fragmentation)
            for fragmentation in fragmentations_str_split:
                [number_groups,number_type_groups] = sum_digits_string(fragmentation)
            
                # If we got a radical, we need to ensure that a radical group will be considered
                if radical_status:
                    num_radicals_in_fragmentation = sum('(.)' in group for group in fragmentation.split(", "))
                    O_center = has_oxygen_radical(smi)
                    if num_radicals_in_fragmentation == 0 or num_radicals_in_fragmentation != num_radicals:
                        # Assign a high number to force this fragmentation to be skipped if no radical group is present
                        number_groups = 8888
                    elif not O_center: # Carbon-centered radical
                        contains_O_fragmentation = any('O(.)' in group for group in fragmentation.split(", "))
                        if contains_O_fragmentation:
                            # Assign a high number to force this fragmentation to be skipped if O-centered radical is present when it shouldn't
                            number_groups = 8888

                        
                # Check for the best group decomposition
                if number_groups < number_groups_final:
                    number_groups_final = number_groups
                    number_type_groups_final = number_type_groups
                    final_fragmentation = fragmentation
                else:
                    if number_groups == number_groups_final and number_type_groups<number_type_groups_final:
                        number_groups_final = number_groups
                        number_type_groups_final = number_type_groups
                        final_fragmentation = fragmentation
                    
    # occurance vector
    groups_UNIFAC = list(fragmentation_scheme.keys())                
    occ = groupocc(final_fragmentation, groups_UNIFAC)
    return final_fragmentation, occ, fragmentations

# TEST
if __name__ == '__main__':
    smiles = 'CCCC(=O)OOOO'
    groups, occ, fragmentations = bestfragmentation(smiles)
    print("final",groups)
    print("all",fragmentations)
    # Draw the molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        plt.imshow(img)
        plt.axis('off')  # Hide axes
        plt.show()
    else:
        print("Invalid SMILES string!")
    