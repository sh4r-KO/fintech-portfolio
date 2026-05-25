import pytest
import json
from BE import app as app_module
from BE.app import compound, CompoundInput


#pytest TESTS/test_app.py -W ignore::UserWarning

class Test_app:
    # --- refresh_data ---
    def test_refresh_data_valid(self,tmp_path,monkeypatch ):
        SAMPLE_PROJECTS = [
            {
                "id": "1", "slug": "test-project", "title": "Test",
                "summary": "A test project", "tags": [], "tech": []
            }, {
                "id": "2", "slug": "test-project2", "title": "Test",
                "summary": "A test project", "tags": [], "tech": []
            }
        ]

        f = tmp_path / "projects.json"
        f.write_text(json.dumps(SAMPLE_PROJECTS))
        monkeypatch.setattr(app_module, "DATA_PATH", f)

        app_module.refresh_data()

        assert len(app_module.PROJECTS) == 2
        assert app_module.PROJECTS[0].slug == "test-project"
        assert app_module.PROJECTS[1].slug == "test-project2"

    def test_refresh_missing_file(self,tmp_path,monkeypatch ):

        f = tmp_path / "-.json"
        monkeypatch.setattr(app_module, "DATA_PATH", f)

        app_module.refresh_data()

        assert app_module.PROJECTS == []
        assert app_module.INDEX_BY_SLUG == {}

    def test_refresh_bad_JSON(self,tmp_path,monkeypatch ):
        f = tmp_path / "projects.json"
        f.write_text(json.dumps("BADJSON"))
        monkeypatch.setattr(app_module, "DATA_PATH", f)

        app_module.refresh_data()

        assert app_module.PROJECTS == []
        assert app_module.INDEX_BY_SLUG == {}

    @pytest.fixture
    def load_projects(self,tmp_path, monkeypatch):
        projects = [
            {"id": "1", "slug": "test-project", "title": "vanilla", "summary": "A test project", "tags": ["a"],
             "tech": []},
            {"id": "2", "slug": "test-project2", "title": "chocolate", "summary": "A test project", "tags": ["b"],
             "tech": []},
            {"id": "3", "slug": "test-project3", "title": "berry", "summary": "A test project", "tags": ["c"],
             "tech": []},
            {"id": "4", "slug": "test-project3", "title": "berry", "summary": "A test project", "tags": ["d"],
             "tech": []},
        ]
        f = tmp_path / "projects.json"
        f.write_text(json.dumps(projects))
        monkeypatch.setattr(app_module, "DATA_PATH", f)
        app_module.refresh_data()

        return len(projects)

    def test_list_projects_no_filters(self,load_projects):
        result = app_module.list_projects()
        assert len(result) == load_projects

    def test_list_projects_filter_by_query(self,load_projects):
        result = app_module.list_projects(q="vanilla")
        assert len(result) == 1
        assert result[0].title == "vanilla"

    def test_list_projects_filter_by_tag(self,load_projects):
        result = app_module.list_projects(tag="b")
        assert len(result) == 1
        assert result[0].tags == ["b"]

    def test_list_projects_filter_by_query_and_tag(self,load_projects):
        result = app_module.list_projects(q="berry", tag="c")
        assert len(result) == 1
        assert result[0].title == "berry"

    #Query that matches nothing — returns empty list
    def test_list_projects_bad_query_return_empty(self,load_projects):
        result = app_module.list_projects(q="aqezdsvqsdfvqsdvqsdvQSDV")
        assert len(result) == 0

    #Tag that doesn't exist — returns empty list
    def test_list_projects_bad_tag_return_empty(self,load_projects):
        result = app_module.list_projects(tag="aqezdsvqsdfvqsdvqsdvQSDV")
        assert len(result) == 0

    #Query matching multiple projects (e.g. all have "A test project" in summary)
    def test_list_projects_query_match_multiple(self,load_projects):
        result = app_module.list_projects(q="berry")
        assert len(result) == 2

    #Case insensitivity — searching "VANILLA" still finds "vanilla"
    def test_list_projects_query_case_sensitive(self,load_projects):
        result = app_module.list_projects(q="VANILLA")
        assert len(result) == 1
        assert result[0].title == "vanilla"

    #limit parameter — ask for 2, get only 2 back
    def test_list_projects_limit(self,load_projects):
        result = app_module.list_projects(limit=2)
        assert len(result) == 2

    #offset parameter — skip the first one, get the rest
    def test_list_projects_offset(self,load_projects):
        result = app_module.list_projects(offset=1)
        assert len(result) == load_projects -1

    #limit and offset combined — pagination basically
    def test_list_projects_offset_with_limit(self,load_projects):
        result = app_module.list_projects(offset=1,limit=2)
        assert len(result) == 2
        assert result[0].id =='2'

    #Offset larger than total projects — returns empty list
    def test_list_projects_offset_too_large(self, load_projects):
        result = app_module.list_projects(offset=load_projects+1)
        assert len(result) == 0

    def test_get_project(self, load_projects):
        result = app_module.get_project(slug="test-project2")
        assert result.id == "2"

    def test_get_project_invalid(self, load_projects):
        with pytest.raises(Exception):
            app_module.get_project(slug="DOENSTEXIT")


    #________________________________________
    #_____________compound TESTS_____________
    #________________________________________

    @pytest.fixture
    def create_payload(self):


        return len(payload)

    def test_compound_(self):
        pass

    def test_compound_no_growth_no_contrib(self):
        payload = CompoundInput(principal=1000, rate_pct=0, years=5, compounds_per_year=12, contribution=0)
        result = app_module.compound(payload)
        assert result.final_value == 1000
        assert result.principal == 1000
        assert result.total_contributions == 0
        assert result.total_interest == 0
        assert len(result.points) == 61

    def test_compound_no_growth_with_contrib(self):
        payload = CompoundInput(principal=0, rate_pct=0, years=3, compounds_per_year=1, contribution=100)
        result = app_module.compound(payload)
        assert result.final_value == 300
        assert result.principal == 0
        assert result.total_contributions == 300
        assert result.total_interest == 0
        assert len(result.points) == 4


#Annual compounding, no contribution
    def test_compound_compounding(self):
        payload = CompoundInput(principal=1000, rate_pct=10, years=2, compounds_per_year=1, contribution=0)
        result = app_module.compound(payload)
        assert result.final_value == 1210
        assert result.principal == 1000
        assert result.total_contributions == 0
        assert result.total_interest == 210
        assert len(result.points) == 3

    def test_compound_compounding(self):
        payload = CompoundInput(principal=1000, rate_pct=10, years=2, compounds_per_year=1, contribution=0)
        result = app_module.compound(payload)
        assert result.final_value == 1210
        assert result.principal == 1000
        assert result.total_contributions == 0
        assert result.total_interest == 210
        assert len(result.points) == 3

    def test_compound_interests_no_principal(self):
        payload = CompoundInput(principal=0, rate_pct=5, years=2, compounds_per_year=1, contribution=1000)
        result = app_module.compound(payload)
        assert result.final_value == 2050
        assert result.principal == 0
        assert result.total_contributions == 2000
        assert result.total_interest == 50
        assert len(result.points) == 3

    def test_compound_no_years(self):
        payload = CompoundInput(principal=5000, rate_pct=10, years=0, compounds_per_year=12, contribution=200)
        result = app_module.compound(payload)
        assert result.final_value == 5000
        assert result.principal == 5000
        assert result.total_contributions == 0
        assert result.total_interest == 0
        assert len(result.points) == 1

    def test_compound_fractionnal_years(self):
        payload = CompoundInput(principal=1000, rate_pct=10, years=0.4, compounds_per_year=4, contribution=200)
        result = app_module.compound(payload)
        assert result.final_value == 1225
        assert result.principal == 1000
        assert result.total_contributions == 200
        assert result.total_interest == 25
        assert len(result.points) == 2


    def test_points_list_length(self):
        payload = CompoundInput(principal=1000, rate_pct=5, years=5, compounds_per_year=12, contribution=0)
        result = app_module.compound(payload)
        assert len(result.points) == 61

    def test_points_structure_consistency(self):
        payload = CompoundInput(principal=1000, rate_pct=10, years=2, compounds_per_year=4, contribution=100)
        result = app_module.compound(payload)
        assert result.points[0].balance == 1000
        assert result.points[0].contributed == 0
        assert result.points[0].interest == 0
        assert result.points[0].t == 0
        for i, p in enumerate(result.points):
            expected_interest = p.balance - p.principal - p.contributed
            assert p.interest == pytest.approx(expected_interest)

    def test_large_values(self):
        payload = CompoundInput(principal=1_000_000_000, rate_pct=15, years=30, compounds_per_year=365, contribution=0)
        result = app_module.compound(payload)
        assert result.final_value == pytest.approx(90_017_131_300.52, rel=1e-2)
        assert result.principal == 1_000_000_000
        assert result.total_contributions == 0
        assert len(result.points) == 10951

    def test_negative_rate_rejected(self):
        with pytest.raises(Exception):
            CompoundInput(principal=1000, rate_pct=-5, years=1, compounds_per_year=1, contribution=0)